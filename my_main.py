from agents.a3c_agent import A3CAgent
import tensorflow as tf

import importlib
import threading
import os
import sys
import time 
import datetime

from absl import app
from absl import flags

from pysc2 import maps
from pysc2.env import available_actions_printer
from run_loop import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
from pysc2.lib import features

FLAGS = flags.FLAGS

# agent/network args
flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")  
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_bool('training', True, 'Whether to train agents.')
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate for agents.')  # name, value, desc
flags.DEFINE_float('gamma', .99, 'discount rate for future rewards.')
flags.DEFINE_integer("max_agent_steps", 120, "Total agent steps.")

# environment args
flags.DEFINE_integer('step_mul', 8, 'Number of steps agent takes per second.')
flags.DEFINE_string("map_name", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_integer('max_steps', int(1e5), 'Max number of steps to run agent.')
flags.DEFINE_integer('screen_res', 32, 'Number of pixels for screen resolution e.g: [n, n].')
flags.DEFINE_integer('minimap_res', 32, 'Number of pixels for minimap resolution e.g: [n, n].')
flags.DEFINE_bool('render', False, 'Whether to render pysc2 interface.')

# logging/other
flags.DEFINE_string('log_path', './log/', 'File path for logging.')
flags.DEFINE_integer('snapshot_step', int(1e2), 'steps between snapshots.')
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")  # snapshot = ckpt file. how to load correct agent?
flags.DEFINE_bool("save_replay", False, "Whether to save replay at the end.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling")  #?
flags.DEFINE_bool("trace", False, "Whether to trace code execution")
flags.DEFINE_integer("parallel", 4, "How many instances to run in parallel.")
flags.DEFINE_string("device", "0", "Device for training.")

FLAGS(sys.argv)

if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

COUNTER = 0
LOCK = threading.Lock()
LOG = FLAGS.log_path + FLAGS.map_name + "/"+datetime.datetime.now().strftime("%Y-%m-%d")[-5:]+"/"
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map_name+"/"
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

# AGENT = A3CAgent(training=True, msize=32, ssize=32)
PLAYERS = [sc2_env.Agent(sc2_env.Race.terran)]

def run_thread(agent, map_name, agent_id=0):
    ''' set up and run sc2_env loop '''
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=FLAGS.map_name,
                step_mul=FLAGS.step_mul,
                visualize=FLAGS.render,
                players=PLAYERS,
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=FLAGS.screen_res, minimap=FLAGS.minimap_res),
                    use_feature_units=True),
                game_steps_per_episode=0
                ) as env:
                env = available_actions_printer.AvailableActionsPrinter(env)

                # Only for a single player!
                # snapshot_dir = SNAPSHOT+str(id)  # if i need later
                replay_buffer = []
                for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
                  if FLAGS.training:
                    replay_buffer.append(recorder)
                    if is_done:
                      counter = 0
                      with LOCK:
                        global COUNTER
                        COUNTER += 1
                        counter = COUNTER
                      # Learning rate schedule
                      learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
                      agent.update(replay_buffer, FLAGS.gamma, learning_rate, counter)
                      replay_buffer = []
                      if counter % FLAGS.snapshot_step == 1:
                        agent.save_model(SNAPSHOT, counter)
                      if counter >= FLAGS.max_steps:
                        break
                  elif is_done:
                    obs = recorder[-1].observation
                    score = obs["score_cumulative"][0]
                    print('Your score is '+str(score)+'!')
                if FLAGS.save_replay:
                  env.save_replay(agent.name)

    except KeyboardInterrupt:
        pass


def main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map_name)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  agents = []
  for i in range(PARALLEL):
    agent = agent_cls(FLAGS.training, FLAGS.minimap_res, FLAGS.screen_res)
    print(f'agent {i} created')  # first is fine, rest will raise error
    agent.build(i > 0, DEVICE[i % len(DEVICE)])  # reuse, dev
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  for i in range(PARALLEL): 
    summary_dir = LOG+str(i)
    summary_writer = tf.summary.FileWriter(summary_dir)  # why just one? when is the summary being saved?
    agents[i].setup(sess, summary_writer)
    print(f'To view Tensorboard, run tensorboard --logdir={summary_dir}')

  agent.initialize()  # is this correct
  if not FLAGS.training or FLAGS.continuation:  # should the training flag be here or just continuation?
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  # Run threads
  threads = []
  for i in range(PARALLEL - 1):
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map_name))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[-1], FLAGS.map_name)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(main)