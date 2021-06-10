import cv2
import ctypes
import numpy as np
from utils.c_utils import load_c_lib, c_ptr, c_int, c_longlong, c_float

class Hash:
    def __init__(self, args):
        self.args = args
        self.hash_lib = load_c_lib('algorithm/hash_lib.cpp')
        self.hash_lib.get_state_tot.restype = ctypes.c_int
        self.hash_lib.get_hash.restype = ctypes.c_longlong
        self.hash_lib.get_return.restype = ctypes.c_float
        self.hash_lib.init(c_int(args.avg_n))

    def score_mask(self, obs):
        obs = obs.copy()
        if self.args.env=='Alien':
            obs[70:81,10:39] = 0
        if self.args.env=='BankHeist':
            obs[67:73,10:80] = 0
        if self.args.env=='BattleZone':
            obs[71:76,40:70] = 0
        if self.args.env=='Frostbite':
            obs[0:15,0:50] = 0
        if self.args.env=='MsPacman':
            obs[74:80,30:60] = 0
        if self.args.env=='RoadRunner':
            obs[0:7,30:60] = 0
        if self.args.env=='StarGunner':
            obs[0:9,30:60] = 0
        if self.args.env=='WizardOfWor':
            obs[5:17,40:80] = 0
        if self.args.env=='Zaxxon':
            obs[0:8,20:60] = 0
        return obs

    def get_hash(self, obs):
        obs = self.score_mask(obs)
        obs = cv2.resize(obs, (42,42), interpolation=cv2.INTER_AREA).astype(np.uint8)//32
        return self.hash_lib.get_hash(c_ptr(obs))

    def get_state_tot(self):
        return self.hash_lib.get_state_tot()

    def get_return(self, h):
        return self.hash_lib.get_return(c_longlong(h))

    def add_return(self, h, value):
        return self.hash_lib.add_return(c_longlong(h), c_float(value))
