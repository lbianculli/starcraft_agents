import importlib
import threading

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

def run_thread(agent_classes, players, map_name, visualize):
    """Run one thread worth of the environment with agents."""
    with sc2_env.SC2Env(map_name=map_name,
                        players=players,
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=FLAGS.feature_screen_size,
                            feature_minimap=FLAGS.feature_minimap_size,
                            rgb_screen=FLAGS.rgb_screen_size,
                            rgb_minimap=FLAGS.rgb_minimap_size,
                            action_space=FLAGS.action_space,
                            use_feature_units=FLAGS.use_feature_units),
                        step_mul=FLAGS.step_mul,
                        game_steps_per_episode=FLAGS.game_steps_per_episode,
                        disable_fog=FLAGS.disable_fog,
                        visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agents = [agent_cls() for agent_cls in agent_classes]
        run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
        if FLAGS.save_replay:
            env.save_replay(agent_classes[0].__name__)
