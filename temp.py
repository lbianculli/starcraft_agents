#!/usr/bin/env python

from pysc2.lib import features, point
#from pysc2 import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common
import ObserverAgent
import importlib
from random import randint
import pickle
from multiprocessing import Process
from tqdm import tqdm
import math
import random
import numpy as np
import multiprocessing
import os
import sys
import logging


class Error(Exception):
    """ Base class for exceptions in this module """
    pass


class ReplayError(Error):
    def __init__(self, message):
        super(ReplayError, self).__init__(message)

def _assert_compat_version(replay_path):
    raise ReplayError(f"Version is incompatible: {replay_path+'.SC2Replay'}")

def _assert_not_corrupt(replay_path):
    raise ReplayError(f"Replay may be corrupt: {replay_path+'.SC2Replay'}")

def _assert_useful(replay_path):
    raise ReplayError(f"Replay not useful for learning purposes. Could be too short or low MMR: {replay_path+'.SC2Replay'}")

def _assert_misc_error(replay_path):
    raise ReplayError(f"Replay could not be loaded: {replay_path+'.SC2Replay'}")


# cpus = multiprocessing.cpu_count() # 16
cpus = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "C:/Users/Program Files (x86)/StarCraft II/Replays/", "Path to the replay files.")
flags.DEFINE_string("agent", "ObserverAgent.ObserverAgent", "Path to an agent.")
flags.DEFINE_integer("procs", cpus, "Number of processes.", lower_bound=1)
flags.DEFINE_integer("frames", 1000, "Frames per game.", lower_bound=1)
flags.DEFINE_integer("start", 0, "Start at replay no.", lower_bound=0)
flags.DEFINE_integer("batch", 16, "Size of replay batch for each process", lower_bound=1, upper_bound=512)
flags.DEFINE_integer("screen_res", 64, "screen resolution in pixels")
flags.DEFINE_integer("minimap_res", 64, "minimap resolution in pixels")
# flags.mark_flag_as_required("replays")
# flags.mark_flag_as_required("agent")
FLAGS(sys.argv)
FILE_OP = None

class Parser:
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 screen_size_px=(64, 64), # (60, 60)
                 minimap_size_px=(64, 64), # (60, 60)
                 discount=1.,
                 frames_per_game=1,
                 logdir="./transform_logger/"):

        # print("Parsing " + replay_file_path)

        self.replay_file_name = replay_file_path.split("/")[-1].split(".")[0]
        # print(f"replay_file_name: {self.replay_file_name}")
        self.agent = agent
        self.discount = discount
        self.frames_per_game = frames_per_game

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(self.replay_file_name + '.SC2Replay')
        ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        # this could be done cleaner i think
        if self._valid_replay(self.info, ping) == "version":
            self.sc2_proc.close()
            _assert_compat_version(self.replay_file_name)
        if self._valid_replay(self.info, ping) == "corrupt":
            self.sc2_proc.close()
            _assert_not_corrupt(self.replay_file_name)
        if self._valid_replay(self.info, ping) == "not_useful":
            self.sc2_proc.close()
            _assert_useful(self.replay_file_name)


        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=64))
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)  # this is working

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)

        self._episode_length = self.info.game_duration_loops
        self._episode_steps = 0
        self._init_logger(logdir)

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """
        Make sure the replay isn't corrupt, and is worth looking at.
        Could I use the below logic to raise varying exceptions?
        """
        if info.HasField("error"):
            return "corrupt"
        if info.base_build != ping.base_build:
            return "version"
        if info.game_duration_loops < 1000 or len(info.player_info) != 2:
            return "not_useful"
        for p in info.player_info:
            if p.player_apm < 60 or (p.player_mmr != 0 and p.player_mmr < 2000):
                return "not_useful"

        return True

    def _init_logger(self, logdir):
      self.logger = logging.getLogger(__name__)
      self.logger.setLevel(logging.INFO)
      file_handler = logging.FileHandler(logdir)
      file_handler.setLevel(logging.INFO)
      formatter = logging.Formatter('%(levelname)s - %(message)s')
      file_handler.setFormatter(formatter)
      self.logger.addHandler(file_handler)

    def start(self):
        # self.logger.info(f"GAME INFO: {self.controller.game_info()}\n")

        aif = features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=FLAGS.screen_res, minimap=FLAGS.minimap_res),
                    use_feature_units=True)
        map_size = (64, 64)
        map_size = point.Point(*map_size)
        # _features = features.Features(self.controller.game_info())  game info not returning info needed
        _features = features.Features(aif, map_size)

        frames = random.sample(np.arange(self.info.game_duration_loops).tolist(), self.info.game_duration_loops)
        # frames = frames[0 : min(self.frames_per_game, self.info.game_duration_loops)]
        step_mul = 10;
        frames = frames[0:int(self.info.game_duration_loops)//step_mul]
        frames.sort()

        last_frame = 0
        i = 0
        # for frame in frames:
        skips = step_mul
        while i < self.info.game_duration_loops:

            # skips = frame - last_fram
            # last_frame = frame
            i += skips
            self.controller.step(skips)
            obs = self.controller.observe()  # error here

            agent_obs = _features.transform_obs(obs) # error here. no attribute observation
            # print("_features working")

            if obs.player_result: # Episode over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += skips

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, obs.actions, self.info, _features)
            if obs.player_result:
                break

            self._state = StepType.MID

        save_path = "C:/Users/lbianculli/agent_replay_data/"+self.replay_file_name+".p"
        with open(save_path, "wb") as f:
            pickle.dump({"info" : self.info, "state" : self.agent.states}, f)

        self.agent.flush()
        self.logger.info("Data successfully saved and flushed")

def parse_replay(replay_batch, agent_module, agent_cls, frames_per_game):
    for replay in replay_batch:
        filename_without_suffix = os.path.splitext(os.path.basename(replay))[0]
        filename = filename_without_suffix + ".p"
        #print(filename)
        if os.path.exists("data_full/"+filename):
            #print('exists continue, ', filename)
            continue

        try:
            parser = Parser(replay, agent_cls(), frames_per_game=frames_per_game)
            parser.start()
        except ReplayError as e:
            print(e)

def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    processes = FLAGS.procs
    replay_folder = FLAGS.replays
    frames_per_game = FLAGS.frames
    batch_size = FLAGS.batch
    for (root, dirs, files) in os.walk(replay_folder, topdown=True):
      if len(files) > 0:
        replays = [root+"/"+replay for replay in files]
        print(f"REPLAY: {replays[1]}")  # all good thru here
    start = FLAGS.start
    try:
        for i in tqdm(range(math.ceil(len(replays)/processes/batch_size))):
            procs = []
            x = i * processes * batch_size
            if x < start:
                continue
            for p in range(processes):
                xp1 = x + p * batch_size
                xp2 = xp1 + batch_size
                xp2 = min(xp2, len(replays))
                # print(replays[xp1])  # seems like still good up to here
                p = Process(target=parse_replay, args=(replays[xp1:xp2], agent_module, agent_cls, frames_per_game))
                p.start()
                procs.append(p)
                if xp2 == len(replays):
                    break
            for p in procs:
                p.join()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # FILE_OP= open("parsed.txt","w+")
    app.run(main)
