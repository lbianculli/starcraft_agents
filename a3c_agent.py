import threading
import tensorflow as tf
from a2c_agent import a2cAgent

# master would coordinate all agents. Model would be a2c network. handling envs seems chllenging,
# esp with window popping up. is it possible w/o VM?
class Master():
  def __init__(self, map_name, lr, save_path, summary_path):
    self.map_name = map_name
    self.opt = tf.train.RMSPropOptimizer(lr, use_locking=True)
    self.save_path = save_path
    self.summary_path = summary_path
    
    
  def train(self):
    

  
  
# this would be individual a2c agent -- have to handle for pushing grads to master
# create env in each instance
class Worker(threading.Thread):
  # set up global variables (class attributes) across different threads
  global_episode = 0
  best_reward = 0
  save_lock = threading.Lock()
  
  def __init__(global_model,
               opt,
               result_queue,
               idx):
    
    self.global_model = global_model
    self.opt = opt
    self.result_queue = result_queue
    self.worker_idx = idx
    
    
  def run(self):
      


  
  
  
 
