import json
import pickle

with open("obs_test.p", "rb") as f:
  obs = pickle.load(f)


print(obs)

test = json.loads(obs)  # if its JSON
