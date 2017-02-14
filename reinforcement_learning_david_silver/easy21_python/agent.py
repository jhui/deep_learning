import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from game21 import *

class Agent:
    def __init__(self, N0=100.0):
        self.Q = np.zeros((11, 22, 2))
        self.N = np.zeros((11, 22, 2))
        self.N0 = N0

    def greedy_policy(self, game):
        e = self.N0 / (self.N0 + np.sum(self.N[game.dealer_sum(), game.player_sum(), :]))
        if (np.random.rand() < e):
            action = ACTION.HIT if np.random.randint(2)==0 else ACTION.STICK
        else:
            action = ACTION.HIT if np.argmax(self.N[game.dealer_sum(), game.player_sum(), :])==ACTION.HIT else ACTION.STICK
        return action

    def play(self):
        game = State()
        status = STATUS.CONTINUE
        visits = []
        rewards = []
        while status == STATUS.CONTINUE:
            action = self.greedy_policy(game)
            pos = 0 if action==ACTION.HIT else 1
            self.N[game.dealer_sum(), game.player_sum(), pos] += 1
            visits.append((game.dealer_sum(), game.player_sum(), pos))
            status, reward = game.step(action)
            rewards.append(reward)
        return visits, rewards