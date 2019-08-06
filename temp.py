      # minimap and screen temps
      m_temp = np.array(state['feature_minimap'], dtype=np.float32)
      m_temp = np.expand_dims(U.preprocess_minimap(m_temp), axis=0)
      s_temp = np.array(state['feature_screen'], dtype=np.float32)
      s_temp = np.expand_dims(U.preprocess_screen(s_temp), axis=0)

      # m_temp = np.array(state['minimap'])
      # m_temp = np.reshape(m_temp, [self.dimension, self.dimension,5])  # this is where shape error is originating
      # s_temp = np.array(state['screen'])
      # s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])

      # one-hot action_id
      last_action = None
      for action in state['actions']:
        if last_action == action:
          continue

        one_hot = np.zeros((1, 543)) # Not sure where 543 is from. Hardcoding for now
        one_hot[np.arange(1), [action[0]]] = 1

        for param in action[2]:
          if param == [0]:  # seen this before. no_op i think
            continue
          minimap_output.append(m_temp)
          screen_output.append(s_temp)
          action_output.append(one_hot[0])
          player_info_output.append(pi_temp)
          ground_truth_coordinates.append(np.array(param))

    assert(len(minimap_output) == len(ground_truth_coordinates))

    if len(minimap_output) == 0:
      # The replay file only record one person's operation, so if it is
      # the defeated person, we need to skip the replay file
      return self.next_batch(get_action_id_only)

    if get_action_id_only:
      return minimap_output, screen_output, player_info_output, action_output
    else:
      return minimap_output, screen_output, action_output, player_info_output, ground_truth_coordinates

