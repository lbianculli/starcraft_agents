import sys
import gin
import numpy as np
from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType
from . import Env, Spec, Space  # Env from /base/.abc

ACTIONS_MINIGAMES, ACTIONS_MINIGAMES_ALL, ACTIONS_ALL = ['minigames', 'minigames_all', 'all']  # "all is full action set"


@gin.configurable
class SCEnv(Env):
  def __init__(
      self,
      map_name="Simple64",  # changed some values here. Think: what else needs to merge this with replay parsing process?
      render=False,
      reset_done=True,
      max_ep_len=None,
      spatial_dim=32,
      step_mul=8,
      obs_features=None,
      action_ids=ACTIONS_ALL):

    super().__init__(map_name, render, reset_done, max_ep_len)

    self.spatial_dim = spatial_dim
    self.step_mul
    self._env = None

    # sensible action set for all minigames
    if not action_ids or action_ids in [ACTIONS_MINIGAMES, ACTIONS_MINIGAMES_ALL]:
      action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]

    # some additional actions for minigames (not necessary to solve)
    if action_ids == ACTIONS_MINIGAMES_ALL:
      action_ids += [11, 71, 72, 73, 74, 79, 140, 168, 239, 261, 264, 269, 274, 318, 335, 336, 453, 477]

    # full action space, including outdated / unusable to current race / usable only in certain cases ** reconcile?
    if action_ids == ACTIONS_ALL:
      action_ids = [f.id for f in actions.FUNCTIONS]

    # by default use majority of obs features, except for some that are unnecessary for minigames
    # e.g. race-specific like creep and shields or redundant like player_id  --> should i do al but player_id?
    if not obs_features:  # ** this will have to be changed
      obs_features = {
          'screen': ['player_relative', 'selected', 'visibility_map', 'unit_hit_points_ratio', 'unit_density'],
          'minimap': ['player_relative', 'selected', 'visibility_map', 'camera'],
          # available actions should always be present and in first position
          'non_spatial': ['available_actions', 'player']}

    self.act_wrapper = ActionWrapper(spatial_dim, action_ids)
    self.obs_wrapper = ObservationWrapper(obs_features, action_ids)

  def start(self):
    """ Creates SC environment """
    from pysc2.env import sc2_env  # lazy load

    # fail-safe if executed not as absl app
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(sys.argv)

    self._env = sc2_env.SC2Env(
        map_name=self.id,  # seems to work in reaver, will it work here? if not just change to self.map_name
        visualize=self.render,
        agent_interface_format=[features.parse_agent_interface_format(
            feature_creen=self.spatial_dim,
            feature_minimap=self.spatial_dim,
            rgb_screen=None,
            rgb_minimap=None)],
        step_mul=self.step_mul
        )

  def step(self, action):  # is this called before or after make_specs()?
    """
    tries to step thru the env. Handles timeout issues.
    If it times out, uses reset to get obs and starts process over.
     """
    try:
      obs, reward, done = self.obs_wrapper(self._env.step(self.act_wrapper(action)))  # a lot going on here
    except protocol.ConnectionError:
      # hack to fix timeout issue
      print("Connection timed out")
      self.restart()
      return self.reset(), 0, 1

    if done and self.reset_done:
      obs = self.reset()

    return obs, reward, done


  def reset(self):
    try:
      obs, reward, done = self.obs_wrapper(self._env.reset())
    except protocol.ConnectionError:
      print("Connection timed out")
      self.restart()
      return self.reset()

    return obs

  def stop(self):
    self._env.close()

  def restart(self):
    self.stop()
    self.start()

  def obs_spec(self):
    """ checks if obs spec exists, otherwise makes it """
    if not self.obs_wrapper.spec:
      self.make_specs()
    return self.obs_wrapper.spec

  def act_spec(self):
    """ checks if act spec exists, otherwise makes it """
    if not self.act_wrapper.spec:
      self.make_specs()
    return self.act_wrapper.spec

  def make_specs(self):
    """ sets up mock env to create specs, then closes """
    from pysc2.env import mock_sc2_env
    mock_env = mock_sc2_env.SC2TestEnv(map_name=self.id, agent_interface_format=[  # again with the self.id
        features.parse_agent_interface_format(feature_screen=self.spatial_dim, feature_minimap=self.spatial_dim)])
    self.act_wrapper.make_spec(mock_env.action_spec())
    self.obs_wrapper.make_spec(mock_env.observation_spec())
    mock_env.close()



class ObservationWrapper:
  """ Takes observation features from API and removes unecessary elements. """
  def __init__(self, _features=None, action_ids=None):
    self.spec = None
    self.features = _features  # dict keyed by screen/mm/non_spatial with list of relevant info from feature_screen/minimap -- from SC2ENV.__init__()
    self.action_ids = action_ids  # list of action ids (idxs) to be sampled from

    screen_feature_to_idx = {feat: idx for idx, feat in enumerate(features.SCREEN_FEATURES._fields)}  # what is _fields?
    minimap_feature_to_idx = {feat: idx for idx, feat in enumerate(features.MINIMAP_FEATURES._fields)}  # think its just another way to set up features dicts

    self.feature_masks = {
    "screen": [screen_feature_to_idx[i] for i in _features["screen"]],  # need to know what _fields looks like to understand...
    "minimap": [minimap_feature_to_idx[i] for i in _features["minimap"]]
    }

  def __call__(self, timestep):
    """ takes timestep from Env to return reward, done and dict of screen/minimap features and available actions """
    ts = timestep[0]
    obs, reward, done = ts.observation, ts.reward, ts.step_type == StepType.LAST

    obs_wrapped = {
        obs["feature_screen"][self.feature_masks["screen"]]
        obs["feature_minimap"][self.feature_masks["minimap"]]
    }

    for feat_name in self.features["non_spatial"]:
      if feat_name == "available_actions":  # filter only for actions we are actually going to use
        fn_id_idxs = [i for i, fn_id in enumerate(self.action_ids) if fn_id in obs[feat_name]]  # gets idx if the function is in obs["available_actions"]
        mask = np.zeros((len(self.action_ids),), dtype=np.int32)
        mask[fn_id_idxs] = 1
        obs[feat_name] = mask
      obs_wrapped.append(obs[feat_name])

    return obs_wrapped, reward, done

  def make_spec(self, spec):  # **Some preprocessing in here.
    """ Final output is a spec of screen, mm, available_action and player spaces """
    spec = spec[0]  # why only 0?

    default_dims = {  # workaround
        "available_actions": len(self.action_ids)
    }

    # tuples of length of relevant feature info
    screen_shape = (len(self.features["screen"]), *spec["feature_screen"][1:])  # where the spec will come from mock_env
    minimap_shape = (len(self.features["minimap"]), *spec["feature_minimap"][1:])  # whats the first element, why are we skipping?
    # list of values relative to features, 1 if scalar, .scale if categorical
    screen_dims = get_spatial_dims(self.features["screen"], features.SCREEN_FEATURES)
    minimap_dims = get_spatial_dims(self.features["minimap"], features.MINIMAP_FEATURES)

    spaces = [
     SCSpace(screen_shape, "screen", self.features["screen"], screen_dims),
     SCSpace(minimap_shape, "minimap", self.features["minimap"], minimap_dims)
     ]

    for feat in self.features["non_spatial"]:  # only available_actions and player. really only pertains to actions though
      if 0 in spec[feat]:
        spec[feat] = default_dims[feat]
      spaces.append(Space(spec[feat], name=feat))

    self.spec = Spec(spaces, "Observation")


class ActionWrapper:
  """ Sets up action id/arg preprocessing and make_spec for action """
  def __init__(self, spatial_dim, action_ids, args=None):
    self.spec = None
    if not args:  # ** need to coordinate this with observer
      args = [
          'screen',
          'minimap',
          'screen2',
          'queued',
          'control_group_act',
          'control_group_id',
          'select_add',
          'select_point_act',
          'select_unit_act',
          # 'select_unit_id'
          'select_worker',
          'build_queue_id',
          # 'unload_id'
      ]

    self.func_ids = action_ids
    self.args, self.spatial_dim = args, spatial_dim

  def __call__(self, action):  # how does __call__ work?
    """ Handles action_id/arg collecting and returns FunctionCall """
    defaults = {  # should these ever not be a 0?
        'control_group_act': 0,
        'control_group_id': 0,
        'select_point_act': 0,
        'select_unit_act': 0,
        'select_unit_id': 0,
        'build_queue_id': 0,
        'unload_id': 0,
    }

    fn_id_idx, args = action.pop(0), []
    fn_id = self.func_ids[fn_id_idx]
    for arg_type in actions.FUNCTIONS[fn_id].args:  # * other preprocessing that looks familiar.
      arg_name = arg_type.name
      if arg_name in self.args:  # e.g: screen, screen2, select_point_add, etc.
        arg = action[self.args.index(arg_name)]  # looks like a way the api allows for getting the id from the name (?)
        # pysc2 expects all args in their separate lists
        if type(arg) not in [list, tuple]:
          arg = [arg]
        # pysc2 expects spatial coords, but we have flattened => attempt to fix ==> (if non-spatial?)
        if len(arg_type.sizes) > 1 and len(arg) == 1:   # i still dont get the point of mod and floor.
          arg = [arg[0] % self.spatial_dim, arg[0] // self.spatial_dim]  # but i think i get the idea.
        args.append(arg)
      else:
        args.append(defaults[arg_name])  # if not in self.args, append a 0  (will this result in no_op?)

    return [actions.FunctionCall(action_id, args)]

  def make_spec(self, spec):
    """ creates Spaces and eventually Spec for each arg """
    spec = spec[0]
    spaces = [SCFuncIdSpace(self.func_ids, self.args)]

    for arg_name in self.args:  # i think the whole Spaces thing is another way of setting up phs for each arg
      arg = getattr(spec.types, arg_name)  # so returns the value of spec.types.arg_name (?)
      if len(arg.sizes) > 1:
        spaces.append(Space(domain=(0, args.sizes), categorical=True, name=arg_name))
      else:
        spaces.append(Space(domain=(0, args.sizes[0]), categorical=True, name=arg_name))

    self.space = Spec(Spaces, "Action")


class SCSpace(space):  # not sure what either of these do exactly. would be good understanding for differences between super() calls
  """ Space object for obs features """
  def __init__(self, shape, name, spatial_feats=None, spatial_dims=None):
    if spatial_feats:
      name += "{%s}" % ", ".join(spatial_feats)
    self.spatial_feats, self.spatial_dims = spatial_feats, spatial_dims

    super().__init__(shape, name=name)


class SCFuncIdSpace(space):
  """ Space object for action features. Takes list of ids and args and returns masked version """
  def __init__(self, func_ids, args):
    super().__init__(domain=(0, len(func_ids)), categorical=True, name="function_id")
    self.args_mask = []
    for fn_id in func_ids:
      fn_id_args = [arg_type.name for arg_type in actions.FUNCTIONS[fn_id].args]
      self.args_mask.append([arg in fn_id_args for arg in args])  # if arg in fn_id_args append to list


def get_spatial_dims(feat_names, feats):
    """
    gets spatial dims of given feature (screen or mm).
    Returns list of those dims with element=1 for scalar, .scale value for categorical
    """
    feat_dims = []
    for feat_name in feat_names:
      feat = getattr(feats, feat_name)
      feat_dims.append(1)
      if feat.type == features.FeatureType.CATEGORICAL:
        feat_dims[-1] = feat.scale

    return feat_dims
