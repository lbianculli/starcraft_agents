import gin.tf
import numpy as np
import tensorflow as tf
from abc import abstractmethod

from reaver.envs.base import Spec
from reaver.agents.base import MemoryAgent
from reaver.utils import Logger  # how does this look?
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType


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
    model_fn: ModelBuilder=None,
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

    self.model = model_fn(obs_sec, act_spec)  # this is where the model build (fully conv) would go, whats obs/act_spec though?
    self.value = self.model.outputs[-1]  # very cool
    self.policy = policy_cls(act_spec, self.model.outputs[:-1])  # whats advantage of doing it like this?
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
    self.logger = Logger()


  def get_action_and_value(self, obs):
    return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)

  def get_action(self, obs):


  def on_step(self, step, obs, action, reward, done, value=None):


  def minimize(self, advantages, returns):


  def compute_advantages_and_returns(self, bootstrap_value):



  def on_start(self):



  def on_finish(self):



  def make_minimize_ops(self):

