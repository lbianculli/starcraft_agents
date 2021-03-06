import os
from os.path import expanduser
import numpy as np
import pickle
from google.protobuf.json_format import MessageToJson
import json
from pysc2.lib import actions
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("C:/Users/lbianculli/action_param_log", mode="w")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class BatchGenerator(object):
	def __init__(self):
		self.home_dir = "C:/Users/lbianculli"
		self.parsed_directory = self.home_dir+'/agent_replay_data/'
		self.parsed_filenames = os.listdir(self.parsed_directory)  # list of pickled replays
		self.next_index = 0
		self.dimension = 64  #

	# every batch corresponding to 1 replay file
	def next_batch(self, get_action_id_only=False):
		""" Loads pickled replay data. Returns screen, minimap, actions, player info and coords. For supervised learning """
		full_filename = self.parsed_directory + self.parsed_filenames[self.next_index]  # single replay path
		# while os.stat(full_filename).st_size == 0:
		while os.path.getsize(full_filename) == 0:
			del self.parsed_filenames[self.next_index]
			full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
			if self.next_index >= len(self.parsed_filenames):
				self.next_index = 0

		self.next_index += 1
		if self.next_index == len(self.parsed_filenames):
			self.next_index = 0

		try:
			replay_data = pickle.load(open(full_filename, "rb"))
		except:
			return self.next_batch(get_action_id_only)

		loaded_replay_info_json = MessageToJson(replay_data['info'])
		info_dict = json.loads(loaded_replay_info_json)

		# get winner from json
		winner_id = -1
		for info in info_dict['playerInfo']:
			if info['playerResult']['result'] == 'Victory':
				winner_id = int(info['playerResult']['playerId'])
				break

		if winner_id == -1:  # if its a tie
			replay_data = [] # release memory
			return self.next_batch(get_action_id_only)

		minimap_output = []  # set up lists for arrays
		screen_output = []
		action_output = []
		player_info_output = []
		ground_truth_coordinates = []

		for state in replay_data['state']:
			if state['actions'] == []:
				continue

			# player info
			info_temp = np.array(state['player'])
			if info_temp[0] != winner_id:
				continue

			# minimap and screen temps
			m_temp = np.array(state['minimap'])
			m_temp = np.reshape(m_temp, [self.dimension, self.dimension,5])  # this is where shape error is originating
			s_temp = np.array(state['screen'])
			s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])
			
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

