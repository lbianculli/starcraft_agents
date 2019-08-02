#!/usr/bin/env python
from pysc2.lib import features, point  # obj for coordinates
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
import ObserverAgent
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common
import importlib
import glob
from random import randint
import pickle
from multiprocessing import Process
import logging
from tqdm import tqdm
import math
import random
import numpy as np
import multiprocessing
import os
import logging
import sys
import json


# cpus = multiprocessing.cpu_count() # 16

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "/home/lbianculli/StarCraftII/Replays", "Path to the replay files.")  # is this correct path?
flags.DEFINE_string("agent", "ObserverAgent.ObserverAgent", "Path to an agent.")
flags.DEFINE_integer("procs", 2, "Number of processes.", lower_bound=1)  # can i default at n-1?
flags.DEFINE_integer("frames", 1000, "Frames per game.", lower_bound=1)
flags.DEFINE_integer("start", 0, "Start at replay no.", lower_bound=0)
flags.DEFINE_integer("batch", 16, "Size of replay batch for each process", lower_bound=1, upper_bound=512)
flags.DEFINE_integer("screen_res", 32, "screen resolution in pixels")
flags.DEFINE_integer("minimap_res", 32, "minimap resolution in pixels")


FLAGS(sys.argv)
assert FLAGS.screen_res == FLAGS.minimap_res

class Parser:
    """  """
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 screen_size_px=(32, 32), # (60, 60)
                 minimap_size_px=(32, 32), # (60, 60)
                 discount=.99,
                 frames_per_game=1,
                 logdir="./transform_replay_logs"):

        print("Parsing " + replay_file_path)

        self.replay_file_name = replay_file_path.split("/")[-1].split(".")[0]
        self.agent = agent
        self.discount = discount
        self.frames_per_game = frames_per_game

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(self.replay_file_name + '.SC2Replay')
        ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        # print(self.info)
        if not self._valid_replay(self.info, ping):
            self.sc2_proc.close()
            # print(self.info)
            raise Exception(f"{self.replay_file_name+".SC2Replay"} is not a valid replay file!")


        self._init_logger(logdir)
        # # screen_size_px = point.Point(*screen_size_px)
        # # minimap_size_px = point.Point(*minimap_size_px)
        # # screen_size_px.assign_to(interface.feature_layer.resolution)
        # # minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)  # could i use this?

        self._episode_length = self.info.game_duration_loops
        self._episode_steps = 0

        # send request to API to start replay
        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST


    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            self.logger.ERROR("InvalidReplayError")  # has to be a more legit way to do this ***
            return False
        for p in info.player_info:
            if p.player_apm < 80 or (p.player_mmr != 0 and p.player_mmr < 3000):

                return False

        return True

    def start(self):
        """ gets features and begins to run agent """
        self.logger.info(f"GAME INFO: {self.controller.game_info()}")

        aif = features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=FLAGS.screen_res, minimap=FLAGS.minimap_res),
                    use_feature_units=True)

        map_size = (32,32)  # there has to be a way to do this in one line
        map_size = point.Point(*map_size)

        # Features class requires agent_interface_format, map_size. Could i not just hardcode that? Could be none to see what happens
        _features = features.Features(aif, map_size)

        frames = random.sample(np.arange(self.info.game_duration_loops).tolist(), self.info.game_duration_loops)
        # frames = frames[0 : min(self.frames_per_game, self.info.game_duration_loops)]
        step_mul = 10;  # why is it 10, literally 3 different values in this thing
        frames = frames[0:int(self.info.game_duration_loops)//step_mul]
        frames.sort()

        last_frame = 0
        i = 0
        # for frame in frames:
        skips = step_mul
        while i < self.info.game_duration_loops:
            # skips = frame - last_frame
            # last_frame = frame
            i += skips
            self.controller.step(skips)
            obs = self.controller.observe()  # is there anyway to get the obs then specifically check which replay it is?
            with open("obs_test.p", "wb") as f:
              pickle.dump(obs, f)

            test = json.loads(obs)
            self.logger.info(f"IF THIS WORKS IM MONEY? {obs.observation}")

    def _init_logger(self, logdir):
      self.logger = logging.getLogger(__name__)
      self.logger.setLevel(logging.ERROR)
      file_handler = logging.FileHandler(logdir)
      file_handler.setLevel(logging.INFO)
      formatter = logging.Formatter('%(levelname)s - %(message)s')
      file_handler.setFormatter(formatter)
      self.logger.addHandler(file_handler)

def parse_replay(replay_batch, agent_module, agent_cls, frames_per_game):  # if i dont need agent_module, why is it here?
    """ Take replay(s) and an agent and try to parse them """
    for replay in replay_batch:
        filename_without_suffix = os.path.splitext(os.path.basename(replay))[0]
        filename = filename_without_suffix + ".p"
        #print(filename)
        if os.path.exists("data_full/"+filename):  # not sure what this is
            #print('exists continue, ', filename)
            continue

        try:
            parser = Parser(replay, agent_cls(), frames_per_game=frames_per_game)
            parser.start()
        except Exception as e:
            print(e)

def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    processes = FLAGS.procs
    replay_folder = FLAGS.replays
    frames_per_game = FLAGS.frames
    batch_size = FLAGS.batch
    start = FLAGS.start

    # replay_folder
    for (root, dirs, files) in os.walk(replay_folder, topdown=True):
      if len(files) > 0:
        replays = [root+"/"+replay for replay in files]
    # print(f"REPLAY: {replays[1]}")

    for i in tqdm(range(math.ceil(len(replays)/processes/batch_size))):
        procs = []
        x = i * processes * batch_size
        if x < start:
            continue
        # would an executor be better here possibly?
        for p in range(processes):
            xp1 = x + p * batch_size
            xp2 = xp1 + batch_size
            xp2 = min(xp2, len(replays))
            p = Process(target=parse_replay, args=(replays[xp1:xp2], agent_module, agent_cls, frames_per_game))
            p.start()
            procs.append(p)
            if xp2 == len(replays):
                break
        for p in procs:
            p.join()

if __name__ == "__main__":

    app.run(main)
