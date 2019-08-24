import gin.tf
import tensorflow as tf

from envs.base import Spec
from utils import StreamLogger
from utils.tensorflow import SessionManager
from utils.typing import ModelBuilder, PolicyType
from .base import SyncRunningAgent, ActorCriticAgent, SupervisedBaseAgent, DEFAULTS


@gin.configurable('A2CAgent')
class SupervisedAgent(SyncRunningAgent, SupervisedBaseAgent):
    """
    A2C: a synchronous version of Asynchronous Advantage Actor Critic (A3C)
    See article for more details: https://arxiv.org/abs/1602.01783
    """
    def __init__(
      self,
      obs_spec: Spec,
      act_spec: Spec,
      model_fn: ModelBuilder=None,
      policy_cls: PolicyType=None,
      sess_mgr: SessionManager=None,
      optimizer: tf.train.Optimizer=None,
      n_envs=4,
      value_coef=DEFAULTS['value_coef'],
      entropy_coef=DEFAULTS['entropy_coef'],
      traj_len=DEFAULTS['traj_len'],
      batch_sz=DEFAULTS['batch_sz'],
      gamma=DEFAULTS['gamma'],
      gae_lambda=DEFAULTS['gae_lambda'],
      clip_rewards=DEFAULTS['clip_rewards'],
      clip_grads_norm=DEFAULTS['clip_grads_norm'],
      normalize_returns=DEFAULTS['normalize_returns'],
      normalize_advantages=DEFAULTS['normalize_advantages'],
    ):
      kwargs = {k: v for k, v in locals().items() if k in DEFAULTS and DEFAULTS[k] != v}

      SyncRunningAgent.__init__(self, n_envs)
      ActorCriticAgent.__init__(self, obs_spec, act_spec, sess_mgr=sess_mgr, **kwargs)
      self.logger = StreamLogger(n_envs=n_envs, log_freq=10, sess_mgr=self.sess_mgr)

    def loss_fn(self):
      """
      Sample trajectories and fit a cost function C. Form grad estimate with C
      and take a TRPO step for next policy.
      """
      adv = tf.placeholder(tf.float32, [None], name="advantages")
      returns = tf.placeholder(tf.float32, [None], name="returns")

      policy_loss = -tf.reduce_mean(self.policy.logli * adv)
      # value_loss = tf.reduce_mean((self.value - returns)**2) * self.value_coef
      value_loss = tf.reduce_mean(self.value - returns)

      entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
      # we want to reduce policy and value errors, and maximize entropy
      # but since optimizer is minimizing the signs are opposite
      full_loss = policy_loss + value_loss - entropy_loss

      return value_loss

      # return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns]  # probably for "logging" purposes

