#!/usr/bin/env python
# this agent is for geting states and actions from replays

import pickle
import numpy as np
import math

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.lib import static_data

FUNCTIONS = actions.FUNCTIONS

class ObserverAgent():

    def __init__(self):
        self.states = []

    def step(self, time_step, actions, info, feat):
        # print(actions)
        state = {}

        state["minimap"] = [
            time_step.observation["feature_minimap"][0] / 255,                  # height_map
            time_step.observation["feature_minimap"][1] / 2,                    # visibility
            time_step.observation["feature_minimap"][2],                        # creep
            time_step.observation["feature_minimap"][3],                        # camera
            (time_step.observation["feature_minimap"][5] == 1).astype(int),       # own_units
            (time_step.observation["feature_minimap"][5] == 3).astype(int),       # neutral_units
            (time_step.observation["feature_minimap"][5] == 4).astype(int),       # enemy_units
            time_step.observation["feature_minimap"][6]                         # selected
        ]

        unit_type = time_step.observation["feature_screen"][6]
        unit_type_compressed = np.zeros(unit_type.shape, dtype=np.float)
        for y in range(len(unit_type)):
            for x in range(len(unit_type[y])):
                if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
                    unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)

        hit_points = time_step.observation["screen"][8]
        hit_points_logged = np.zeros(hit_points.shape, dtype=np.float)
        for y in range(len(hit_points)):
            for x in range(len(hit_points[y])):
                if hit_points[y][x] > 0:
                    hit_points_logged[y][x] = math.log(hit_points[y][x]) / 4  # what is the idea behind this?

        state["screen"] = [
            time_step.observation["feature_screen"][0] / 255,               # height_map
            time_step.observation["feature_screen"][1] / 2,                 # visibility
            time_step.observation["feature_screen"][2],                     # creep
            time_step.observation["feature_screen"][3],                     # power
            (time_step.observation["feature_screen"][5] == 1).astype(int),  # own_units
            (time_step.observation["feature_screen"][5] == 3).astype(int),  # neutral_units
            (time_step.observation["feature_screen"][5] == 4).astype(int),  # enemy_units
            unit_type_compressed,                                   # unit_type
            time_step.observation["feature_screen"][7],                     # selected
            hit_points_logged,                                      # hit_points
            time_step.observation["feature_screen"][9] / 255,               # energy
            time_step.observation["feature_screen"][10] / 255,              # shields
            time_step.observation["feature_screen"][11]                     # unit_density
        ]

        # Binary encoding of available actions
        '''
        state["game_loop"] = time_step.observation["game_loop"]
        '''
        state["player"] = time_step.observation["player"]

        state["available_actions"] = np.zeros(len(sc_action.FUNCTIONS))
        for i in time_step.observation["available_actions"]:
            state["available_actions"][i] = 1.0

        transformed_actions = []
        for a in actions:
            try:
                pysc2_function_call = feat.reverse_action(a)
                func_id = pysc2_function_call.function
                func_name = FUNCTIONS[func_id].name
                func_args = pysc2_function_call.arguments
                transformed_actions.append([func_id, func_name, func_args])
            except:
                pass
                print(e)

        state["actions"] = transformed_actions

        # state["actions"] = actions
        self.states.append(state)


    def flush(self):
      self.states = []
