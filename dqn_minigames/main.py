import importlib
import threading

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
from pysc2.env import run_loop

from dqn_move_only import DQNMoveOnlyAgent as DQNAgent

            
def run_loop(agents, players, map_, visualize=False):  # could i incorporate this elsewhere?
    ''' set up and run sc2_env loop '''
    agent = DQNAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_,
                step_mul=8,
                visualize=visualize,
                players=players,
                agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=32, minimap=32), use_feature_units=True),
                game_steps_per_episode=0
                ) as env:

                env = available_actions_printer.AvailableActionsPrinter(env)  # what this do?
                run_loop.run_loop(agent, env, max_frames=0, max_episodes=0)

                
    except KeyboardInterrupt:
        pass
    
    
def main(unused_argv):
    ''' run agent in env loop '''
    stopwatch.sw.enabled = False
    stopwatch.sw.trace = False
    
    map_ = 'CollectMineralShards'
    map_inst = maps.get(map_)
    agent = DQNAgent()
    players=[sc2_env.Agent(sc2_env.Race.terran)]
    
    run_loop(agent_classes, players, map_, visualize=True)


if __name__ == '__main__':
    app.run(main)
    
