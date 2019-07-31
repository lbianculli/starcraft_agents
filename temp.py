#!/usr/bin/env python

from pysc2.lib import features, point  # obj for coordinates
#from pysc2 import features, point
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as sc_common
import importlib
import glob
from random import randint
import pickle
from multiprocessing import Process
from tqdm import tqdm
import math
import random
import numpy as np
import multiprocessing
import os

# cpus = multiprocessing.cpu_count() # 16
cpus = 8 # only use 8 cpu cores

FLAGS = flags.FLAGS
flags.DEFINE_string("replays", "C:/users/lbianculli/Documents/Replays/Multiplayer/", "Path to the replay files.")  # is this correct path?
flags.DEFINE_string("agent", None, "Path to an agent.")
flags.DEFINE_integer("procs", cpus, "Number of processes.", lower_bound=1)
flags.DEFINE_integer("frames", 1000, "Frames per game.", lower_bound=1)
flags.DEFINE_integer("start", 0, "Start at replay no.", lower_bound=0)
flags.DEFINE_integer("batch", 16, "Size of replay batch for each process", lower_bound=1, upper_bound=512)
flags.DEFINE_INTEGER("screen_res", "screen resolution in pixels", 32)
flags.DEFINE_INTEGER("minimap_res", "minimap resolution in pixels", 32)
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("agent")

FILE_OP = None

class Parser:
    """  """
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 screen_size_px=(64, 64), # (60, 60)
                 minimap_size_px=(64, 64), # (60, 60)
                 discount=1.,
                 frames_per_game=1):

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
            raise Exception("{} is not a valid replay file!".format(self.replay_file_name + '.SC2Replay'))

        # global FILE_OP
        # FILE_OP.write(self.replay_file_name + '.SC2Replay')

        # self.replay_file_name = self.info.map_name+'_'+self.replay_file_name
        # for player_info in self.info.player_info:
        #     race = sc_common.Race.Name(player_info.player_info.race_actual)
        #     self.replay_file_name = race + '_' + self.replay_file_name


        # screen_size_px = point.Point(*screen_size_px)
        # minimap_size_px = point.Point(*minimap_size_px)
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=FLAGS.screen_res, minimap=FLAGS.minimap_res),
            use_feature_units=True)
        # screen_size_px.assign_to(interface.feature_layer.resolution)
        # minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)

        self._episode_length = self.info.game_duration_loops
        self._episode_steps = 0

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
            return False
        for p in info.player_info:
            # print(p.player_mmr)
            if p.player_apm < 60 or (p.player_mmr != 0 and p.player_mmr < 2000):
                # Low APM = player just standing around.
                # Low MMR = corrupt replay or player who is weak.
                return False

        return True

    def start(self):
        """ gets features and begins to run agent """
        _features = features.Features(self.controller.game_info())

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
            # skips = frame - last_frame
            # last_frame = frame
            i += skips
            self.controller.step(skips)
            obs = self.controller.observe()
            agent_obs = _features.transform_obs(obs.observation)

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

        print("Saving data")
        # save info, staets to pickle file
        pickle.dump({"info" : self.info, "state" : self.agent.states}, open("data/" + self.replay_file_name + ".p", "wb"))
        print("Data successfully saved")
        self.agent.flush()
        print("Data flushed")

        print("Done")

def parse_replay(replay_batch, agent_module, agent_cls, frames_per_game):
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
    print(f"REPLAY: {replays[1]}")

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
    # FILE_OP= open("parsed.txt","w+")
    app.run(main)
