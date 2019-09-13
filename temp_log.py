

---------------------------------

import bs4 as bs
import os
import requests

headers = {
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/72.0.3626.81 Safari/537.36'
    }


class FilingScraper:
    def __init__(self, ticker):
        self.ticker = ticker
        self.file_htms = []


    def get_urls(self):
        url = f"http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={self.ticker}&type=&dateb=&owner=&start=1&count=100&output=xml"
        sec_page = requests.get(url, headers=headers, timeout=5).text  # make sure headers is correct on comp
        sec_soup = bs.BeautifulSoup(sec_page)

        for f in sec_soup.find_all("filing"):
            if f.find("type").get_text() == "10-Q" or f.find("type").get_text() == "10-K":
                hrefs.append(f.find("filinghref").get_text())

        for url in hrefs:
            archive_page = requests.get(url, headers=headers, timeout=5).text
            archive_soup = bs.BeautifulSoup(archive_page)

            table = archive_soup.find("table", {"class": "tableFile"})
            for tr in table.tbody.find_all("tr")[1:]:
                if tr.find_all("td")[2] == "10-Q" or tr.find_all("td")[2] == "10-K":  # clean this and above
                        # feel like i need something else here. smthn to do with the href i think. pretty sure its in the docs
                        self.file_htms.append("https://www.sec.gov" + tr.find("a"))
            # once this block is all set its just parsing the actual files. more of the same

    def parse(self):
        for f in file_htms:
            return _parse(f)

    def _parse(self, url):

-------------------------

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
    
------------------------------ 
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


