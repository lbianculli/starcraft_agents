import gin.tf
import numpy as np
import tensorflow as tf
from abc import abstractmethod

from envs.base import Spec
from agents.base import MemoryAgent
from utils import Logger  # how does this look?
from utils.tensorflow import SessionManager
from utils.typing import ModelBuilder, PolicyType


DEFAULTS = dict(  # will have to change a lot of this later
    model_fn=None,
    policy_cls=None,
    optimizer=None,
    learning_rate=5e-4,
    value_coef=0.5,
    entropy_coef=0.01,
    traj_len=16,
    batch_sz=16,
    gamma=0.99,
    gae_lambda=0.95,
    clip_rewards=0.0,
    clip_grads_norm=0.0,  # what are good values for these?
    normalize_returns=False,
    normalize_advantages=False,
)

@gin.configurable('ACAgent')
class ActorCriticAgent(MemoryAgent):
  """
  Abstract class, unifies deep actor critic functionality
  Handles on_step callbacks, either updating current batch
  or executing one training step if the batch is ready

  Extending classes only need to implement loss_fn method
  """
  def __init__(
    self,
    obs_spec: Spec,  # how does this just work like that?
    act_spec: Spec,
    model_fn: ModelBuilder=None,  # same with these. Has to do with gin and/or __init__ files
    policy_cls: PolicyType=None,
    sess_mgr: SessionManager=None,
    optimizer: tf.train.Optimizer=None,
    value_coef=DEFAULTS['value_coef'],
    entropy_coef=DEFAULTS['entropy_coef'],
    traj_len=DEFAULTS['traj_len'],
    batch_sz=DEFAULTS['batch_sz'],
    gamma=DEFAULTS['gamma'],
    gae_lambda=DEFAULTS['gae_lambda'],
    clip_rewards=DEFAULTS['clip_rewards'],
    clip_grads_norm=DEFAULTS['clip_grads_norm'],
    normalize_returns=DEFAULTS['normalize_returns'],
    normalize_advantages=DEFAULTS['normalize_advantages']):
    MemoryAgent.__init__(self, obs_spec, act_spec, traj_len, batch_sz)

    if not sess_mgr:
      sess_mgr = SessionManager()
    if not optimizer:
      optimizer = tf.train.AdamOptimizer(DEFAULTS["learning_rate"])

    self.sess_mgr = sess_mgr
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.traj_len = traj_len
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.clip_rewards = clip_rewards
    self.normalize_returns = normalize_returns
    self.normalize_advantages = normalize_advantages

    self.model = model_fn(obs_sec, act_spec)  # this is fully_conv
    self.value = self.model.outputs[-1]  # very cool
    self.policy = policy_cls(act_spec, self.model.outputs[:-1])  # whats advantage of doing it like this? Also Policy(act_spec, logits)
    self.loss_op, self.loss_terms, self.loss_inputs = self.loss_fn()  # does this have to change?

    grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
    self.grads_norm = tf.global_norm(grads)
    if clip_grads_norm > 0.:  # currently defaults at 0.
      grads, _  = tf.clip_by_global_norm(grads, clip_grads_norm, self.grads_norm)
    self.train_op = optimizer.apply_gradients(zip(grads, vars), global_step=sess_mgr.global_step)
    self.minimize_ops = self.make_minimize_ops()  # whats benefit of doing like this instead of just setting them in the method?

    sess_mgr.restore_or_init()
    self.n_batches = sess_mgr.start_step
    self.start_step = sess_mgr.start_step * traj_len  # why is there no self on these?
    self.logger = Logger()  # whats this look like exactly?


  def get_action_and_value(self, obs):
    """ feed model.inputs to get policy sample and value from obs """
    return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)  # samples from each dist in space

  def get_action(self, obs):
    """ just get policy sample """
    return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)  #self.model.inputs = [screen, mm] + non+spatial; (op, tf_inputs, inputs)

  def on_step(self, step, obs, action, reward, done, value=None):
    """  
    Note that here "step" refers to update step, rather than agent timestep
    """
    MemoryAgent.on_step(self, step, obs, action, reward, done, value)  # inherit on_step from memoryagent
    self.logger.on_step(self, reward, done)  #?

    if not self.batch_ready():
      return

    next_value = self.sess_mgr.run(self.value, self.model.inputs, self.last_obs)
    adv, returns = self.compute_advantages_and_returns(next_value)

    loss_terms, grads_norm = self.minimize(adv, returns)

    self.sess_mgr.on_update(self.n_batches)
    self.logger.on_update(self.n_batches, loss_terms, grads_norm, returns, adv, next_values)

  def minimize(self, advantages, returns):
    """ feeds advs and returns thru minimize ops to get loss_terms and norm """
    inputs = self.obs + self.acts + [advantages, returns]  # remember: obs and acts are just arrays whose size depends on env
    inputs = [a.reshape(-1, *a.shape[:2]) for a in inputs]  # aligns inputs to exclude first 2 values
    tf_inputs = self.model.inputs + self.policy.inputs + self.loss_inputs

    loss_terms, grads_norm, *_ = self.sess_mgr.run(self.minimize_ops, tf_inputs, inputs)

    return loss_terms, grads_norm

  def compute_advantages_and_returns(self, bootstrap_value):
    """
    Computes discounted returns and advantages. Utilizes GAE and normalization if specified in self.
    GAE can help with reducing variance of policy gradient estimates
    """
    if self.clip_rewards > 0.0:  # (a, min, max, out)
      np.clip(self.rewards, -self.rewards, self.clip_rewards, out=self.rewards)  # clips rewards based on min/max and returns 

    rewards = self.rewards.copy()
    rewards[-1] += (1 - self.dones[-1]) * self.gamma * bootstrap_value  # get most recent reward
    masked_discounts = self.gamma * (1 - self.dones)
    returns = self.discounted_cumsum(rewards, masked_discounts)

    if self.gae_lambda > 0.0:
      values = np.append(self.values, np.expand_dims(bootstrap_value, 0), axis=0)
      # d_t = r_t + g * V(s_{t+1}) - V(s_t)
      deltas = self.rewards + masked_discounts * values[1:] - returns[:-1]
      adv = self.discounted_cumsum(deltas, self.gae_lambda * masked_discounts)  # should look back over this paper
    else:  
      adv = returns - self.values  # self.values instantiated as empty array in MemoryAgent

    if self.normalize_advantages:
      adv = (adv - adv.mean()) / (adv.std() + 1e-10)
    if self.normalize_returns:
      returns = (returns - returns.mean()) / (returns.std() + 1e-10)

    return adv, returns

  def on_start(self):  # not sure what these next two do. 
    self.logger.on_start()

  def on_finish(self):
    self.logger.on_finish()

  def make_minimize_ops(self):
    """ returns loss, norm, and training ops """
    ops = [self.loss_terms, self.grads_norm]
    if self.sess_mgr.training_enabled:
      ops.append(self.train_op)
    # appending extra model update ops (e.g. running stats)
    # note: this will most likely break if model.compile() is used
    ops.extend(self.model.get_updates_for(None))  # get_updates_for is keras.Model method. what does this line do?
    return ops

  @staticmethod
  def discounted_cumsum(x, discount):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    # e.g for reward in range(rewards-2), going backwards from the last reward --> except first two rewards (?)
    for t in range(x.shape[0]-2, -1, -1):  # why -2?
      y[t] = x[t] + discount[t] * y[t+1]
  
    return y

  @abstractmethod
    def loss_fn(self): ...
