from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import pickle
import numpy as np
import pandas as pd
import logging
import os.path
import math
import random


NO_OP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
SELECT_ARMY = actions.FUNCTIONS.select_army.id
ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

FUNCTIONS = actions.FUNCTIONS
DO_NOTHING = 'donothing'
BUILD_DEPOT = 'builddepot'
BUILD_BARRACKS = 'buildbarracks'
BUILD_MARINE = 'buildmarine'
ATTACK = 'attack'

smart_actions = [
    DO_NOTHING,      # 0
    BUILD_DEPOT,     # 1
    BUILD_BARRACKS,  # 2
    BUILD_MARINE,    # 3

]

# split [simple64] into four quadrants
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

killed_unit_reward = .1
killed_building_reward = .5


class QLearningTable():
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, greedy=0.9):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) 
        self.disallowed_actions = {}


    def choose_action(self, observation, excluded_actions=[]):
        '''
        Check if state exists. Choose best action accordingly. If the random
        value is less than epsilon, randomly explore actions instead.

        '''

        self.check_state_exists(observation)
        self.disallowed_actions[observation] = excluded_actions

        state_action = self.q_table.loc[observation, :]
        for excluded_action in excluded_actions:  
            del state_action[excluded_action]
        if np.random.uniform() < self.epsilon:
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()

        else:
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s1, action, reward, s2):
        '''
        checks if both states exist, then comptes target and prediction to solve equation
        and update the table with the value for given s,a pair
        '''
        if s1 == s2:
            return

        self.check_state_exists(s2)
        self.check_state_exists(s1)

        q_predict = self.q_table.loc[s1, action]  
        state_rewards = self.q_table.loc[s2, :]

        if s2 in self.disallowed_actions:  # filter invalid actions
            for excluded_action in self.disallowed_actions[s2]:
                del state_rewards[excluded_action]

        if s2 != 'terminal':
            q_target = reward + self.gamma * state_rewards.max()
        else:
            q_target = reward

        # update
        self.q_table.loc[s1, action] += self.alpha * (q_target - q_predict)


    def check_state_exists(self, state):
        if state not in self.q_table.index:
            # append new state to Q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                               index=self.q_table.columns, name=state))


class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None

        self.main_base_x = None
        self.main_base_y = None
        self.initial_build_complete = False
        self.move_number = 0
        self.initial_build_number = 0
        self.supply_num = 0
        self.scvs_on_gas = 0
        self.build_number_gas = 0

        # if file exists, load the learning table from it. Preserves learning history
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and  # if we have a single item selected and it's a larva
            obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        else:
            return False


    def get_units(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def transform_location(self, obs, x, y, mm_x=64, mm_y=64):
        ''' return new x,y coords based on dims of minimap, starting spawn '''
        if not self.base_top_left:
            return mm_x - x, mm_y - y

        else:
            return x, y


    def transform_distance(self, obs, x, x_dist, y, y_dist):
        ''' taking x and y as input, returns new coords that are of input
            distance from those points. Based on minimap dims'''

        if not self.base_top_left:
            x = x - x_dist
            y = y - y_dist

        else:
            x = x + x_dist
            y = y + y_dist

        return (x, y)

    def split_action(self, obs, action_id):
        ''' Returns action with x, y coords if applicable '''
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def random_camera_loc(self, obs, return_main=False):
        ''' Centers camera around random coordinates, or returns camera to main '''

        if return_main == True:
            FUNCTIONS.move_camera((self.player_x, self.player_y))

        x_trans = random.randint(1, 5)
        y_trans = random.randint(1, 5)

        mm_loc = self.transform_distance(obs, self.player_mm_x, x_trans, self.player_mm_y, y_trans)
        return FUNCTIONS.move_camera(mm_loc)


    def step(self, obs):  # obs is the game observation -- feature maps
        super(SmartAgent, self).step(obs)

        if obs.last():
            print('last')
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            # 'reset' the agent
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0

            return FUNCTIONS.no_op()

        if obs.first():
            # instantiate own main base coords on first step
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            self.player_mm_x = player_x.mean()
            self.player_mm_y = player_y.mean()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
            self.main_base_x = (self.get_units(obs, units.Terran.CommandCenter)[0]).x
            self.main_base_y = (self.get_units(obs, units.Terran.CommandCenter)[0]).y
            self.main_geyser = self.get_units(obs, units.Neutral.VespeneGeyser)[0]  # (48,46) TL, (35,37) BR
            self.idle_selected = 0
            self.main_mins = self.get_units(obs, units.Neutral.MineralField)
            logger.debug(f'({self.main_base_x, self.main_base_y})')


        barracks = self.get_units(obs, units.Terran.Barracks)
        cc_count = len(self.get_units(obs, units.Terran.CommandCenter))
        depot_count = len(self.get_units(obs, units.Terran.SupplyDepot))
        barracks_count = len(barracks)
        current_supply = obs.observation.player.food_used
        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_army
        worker_supply = obs.observation.player.food_workers
        free_supply = supply_limit - current_supply
        scvs = self.get_units(obs, units.Terran.SCV)
        ccs = self.get_units(obs, units.Terran.CommandCenter)
        refinery_count = len(self.get_units(obs, units.Terran.Refinery))
        factories = self.get_units(obs, units.Terran.Factory)
        orbitals = self.get_units(obs, units.Terran.OrbitalCommand)

        if len(scvs) > 1:
            scv_main1 = scvs[0]
            scv_main2 = scvs[1]
        barracks_per_base = math.floor(2.5 * cc_count) #2, 5, 7, etc


        if self.initial_build_complete == False:
            if len(scvs) < 14:
                if not self.is_selected(obs, units.Terran.CommandCenter):
                    return FUNCTIONS.select_point('select', (self.main_base_x, self.main_base_y))

                else:
                    if FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions:
                        return FUNCTIONS.Train_SCV_quick('queued')


            elif not self.is_selected(obs, units.Terran.SCV):
                if depot_count == 0:
                    return FUNCTIONS.select_point('select', (scv_main1.x, scv_main1.y))

            elif self.is_selected(obs, units.Terran.SCV) and depot_count == 0:
                if FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions:

                    target = self.transform_distance(obs, self.main_base_x, 15, self.main_base_y, -9)
                    return FUNCTIONS.Build_SupplyDepot_screen('now', target)

            elif barracks_count == 0:
                if FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:

                    target = self.transform_distance(obs, self.main_base_x, 15, self.main_base_y, 20)
                    return FUNCTIONS.Build_Barracks_screen('now', target)

            elif barracks_count == 1 and self.initial_build_number < 3:  # select scv and build refinery
                if self.initial_build_number == 0:
                    self.initial_build_number += 1
                    return FUNCTIONS.select_point('select', (scv_main2.x, scv_main2.y))

                elif FUNCTIONS.Build_Refinery_screen.id in obs.observation.available_actions and self.initial_build_number == 1:
                    self.initial_build_number += 1
                    return FUNCTIONS.Build_Refinery_screen('now', (self.main_geyser.x, self.main_geyser.y))

                # getting other 2 scvs in gas
                elif self.initial_build_number == 2 and refinery_count > 0:
                    # logger.debug(f'build_num: {self.build_number_gas}')
                    refinery = self.get_units(obs, units.Terran.Refinery)[0]
                    gas_scvs = random.sample(scvs, 2)

                    if self.build_number_gas == 0:
                        self.build_number_gas += 1
                        return FUNCTIONS.select_point('select', (gas_scvs[0].x, gas_scvs[0].y))

                    elif self.build_number_gas == 1:
                        self.build_number_gas += 1
                        return FUNCTIONS.Harvest_Gather_screen('now', (refinery.x, refinery.y))

                    elif self.build_number_gas == 2:
                        self.build_number_gas += 1
                        return FUNCTIONS.select_point('select', (gas_scvs[1].x, gas_scvs[1].y))

                    elif self.build_number_gas == 3:
                        self.build_number_gas += 1
                        self.initial_build_number = 3
                        return FUNCTIONS.Harvest_Gather_screen('now', (refinery.x, refinery.y))


            ### RIGHT HERE!! Need to figure out how to check if specific unit is selected with .is_selected
            if self.initial_build_number == 3 and barracks[0].build_progress == 100:
                if obs.observation.player.idle_worker_count > 0:
                    return FUNCTIONS.Harvest_Gather_screen('now', (self.main_mins[0].x, self.main_mins[0].y))

                if obs.observation.player.idle_worker_count == 0 and not self.is_selected(obs, units.Terran.CommandCenter):

                    return FUNCTIONS.select_point('select', (self.main_base_x, self.main_base_y))

                elif FUNCTIONS.Morph_OrbitalCommand_quick.id in obs.observation.available_actions:

                    self.initial_build_number += 1
                    return FUNCTIONS.Morph_OrbitalCommand_quick('now')


            if self.initial_build_number == 4:  # make reaper
                rax = barracks[0]
                if not self.is_selected(obs, units.Terran.Barracks):
                    return FUNCTIONS.select_point('select', (rax.x, rax.y))

                elif FUNCTIONS.Train_Reaper_quick.id in obs.observation.available_actions:
                    self.initial_build_number += 1
                    return FUNCTIONS.Train_Reaper_quick('now')

            if self.initial_build_number == 5 and FUNCTIONS.Build_Reactor_quick.id in obs.observation.available_actions:
                self.initial_build_complete
                rax = barracks[0]
                self.initial_build_complete = True
                return FUNCTIONS.Build_Reactor_quick('now')  # think want this vs Build_Reactor_quick


        # ---------------------------------------------#
        elif self.initial_build_complete == True:
            if obs.observation.player.idle_worker_count > 0:
                if self.idle_selected == 0:
                    self.idle_selected += 1
                    return FUNCTIONS.select_idle_worker('select_all')

                if self.idle_selected == 1:
                    self.idle_selected = 0
                    return FUNCTIONS.Harvest_Gather_screen('now', (self.main_mins[0].x, self.main_mins[0].y))
                  
        return FUNCTIONS.no_op()
      

def main(unusedargv):
    agent = TerranAgent()
    map = 'Simple64'
    try:
        while True:  
            with sc2_env.SC2Env(
                map_name=map,
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.terran,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat( 
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True), 
                step_mul=25, 
                game_steps_per_episode=0, 
                visualize=True  # True gives you a full version of the visuals
                ) as env:

                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():  
                        break
                    timesteps = env.step(step_actions)  

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
