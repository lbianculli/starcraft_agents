import os
import multiprocessing
import tensorflow as tf
import math

from tqdm import tqdm # tqdm just progress bar decorator
from absl import app, flags
from test_env import ControllerEnv
from utils import clean_dir
# ** do i want two funcs/cls or is it clean enough in main?

class MultiprocEnv:
    """ I still feel like theres a way to make this flow easier """
    def __init__(
        self,
        replay_path="C:/Program Files (x86)/StarCraft II/Replays/",
        processes=8,
        batch_size=16,
        delete_files=False):

        if processes == -1:
            processes = multiprocessing.cpu_count()

        self.processes = processes
        self.batch_size = batch_size
        self.delete_files = delete_files
        self.replays = [replay_path+f for f in os.listdir(replay_path)]
        self.files_to_clean = []

        assert len(self.replays) > 0

    def run(self):
        try:
            for i in tqdm(range(math.ceil(len(self.replays)/self.processes/self.batch_size))):
                procs = []
                x = i * self.processes * self.batch_size
                if x < 0:
                    continue

                for p in range(self.processes):
                    batch_start = x + p * self.batch_size
                    batch_end = batch_start + self.batch_size
                    batch_end = min(batch_end, len(self.replays))
                    p = multiprocessing.Process(target=self._run, args=([self.replays[batch_start: batch_end]]))
                    p.start()
                    procs.append(p)
                    if batch_end == len(self.replays):
                        clean_dir.move_files_list(self.files_to_clean, self.delete_files)
                        break
                for p in procs:
                    p.join()

        except Exception as e:
            print("******", e, "******")

    def _run(self, replay_batch):
        for replay in replay_batch:
            try:
                env = ControllerEnv(replay)
                env.start()
                env.run()

            except KeyboardInterrupt:
                env.controller.quit()
            except Exception as e:
                print(e)
                self.files_to_clean.append(replay)



def main(argv):

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    env = MultiprocEnv()
    env.run()

if __name__ == '__main__':
    app.run(main)
    
    
import os
import shutil


from_dir = "C:/Program Files x86/StarCraft II/Replays/"  # make sure this exact
to_dir = "C:/Users/lbianculli/"

def move_files_dir(from_dir, to_dir, delete_files=False):  # how can i incorporate this into main
    """
    Either moves files from one directory to another or deletes files entirely.
    Make sure to include trailing '/'
    """
    if not os.path.exists(to_dir) and (delete_files == False):
        os.makedirs(to_dir)
    for f in os.listdir(from_dir):
        if delete_files == True:
            os.remove(from_dir+f)
        else:
            shutil.move(from_dir+f, to_dir+f)

def move_files_list(file_list, delete_files=False):
    """ file_list requires files with full path """
    if not os.path.exists(to_dir) and (delete_files == False):
        os.makedirs(to_dir)
    for f in file_list:
        if delete_files == True:
            os.remove(f)
        else:
            shutil.move(f, to_dir+f)


