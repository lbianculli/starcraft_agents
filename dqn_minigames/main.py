import importlib
import threading

from absl import flags, app

from pysc2 import maps
from pysc2.env import run_loop as run_loop
from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import stopwatch, actions, features, units, point_flag
from dqn_move_only import DQNMoveOnlyAgent as DQNAgent


AGENT = DQNAgent(learning_rate=5e-4,  # note, epsilon follows a schedule in DQN homework...linear schedule?
                        batch_size=1,
                        prioritized=True)


def run_thread(agents, players, map_, visualize=False):
    ''' set up and run sc2_env loop '''
    # agents = [AGENT,]
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_,
                step_mul=8,
                visualize=visualize,
                players=players,
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen=32,
                    feature_minimap=32,
                    action_space=None,  # what to do about this later?
                    use_feature_units=True),
                game_steps_per_episode=0
                ) as env:


                # agents = [agent_cls() for agent_cls in agent_classes]
                env = available_actions_printer.AvailableActionsPrinter(env)  # what this do?
                run_loop.run_loop(agents, env, max_frames=0, max_episodes=0)


    except KeyboardInterrupt:
        pass


def main(unused_argv):
    ''' run agent in env loop '''
    stopwatch.sw.enabled = False
    stopwatch.sw.trace = False

    map_ = 'CollectMineralShards'
    map_inst = maps.get(map_)
    agents = [AGENT,]


    players=[sc2_env.Agent(sc2_env.Race.terran)]

    run_thread(agents, players, map_, visualize=False)


if __name__ == '__main__':
    app.run(main)




