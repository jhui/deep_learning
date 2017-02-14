import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from game21 import *
from agent import *

class SarsaApproximator:
    dealer_ranges = np.array([[1, 4], [4, 7], [7, 10]])
    player_ranges = np.array([[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]])

    def __init__(self, lmbda=0.0, N0=100.0):
        self.lmbda = lmbda
        self.N0 = N0
        self.w = np.random.randn(SarsaApproximator.dealer_ranges.shape[0] * SarsaApproximator.player_ranges.shape[0] * 2)

    def action_value_function(self, features):
        return np.sum(self.w * features)

    def greedy_policy(self, hit_value, stick_value):
        e = 0.05
        if (np.random.rand() < e):
            action = ACTION.HIT if np.random.randint(2)==0 else ACTION.STICK
        else:
            action = ACTION.HIT if hit_value>stick_value else ACTION.STICK
        return action

    def get_feature(self, ranges, value):
        result = []
        for i, f_range in enumerate(ranges):
            if f_range[0] <= value <= f_range[1]:
                result.append(i)
        return result

    def featureize(self, dealer_sum, player_sum, action):
        dealer_index = self.get_feature(SarsaApproximator.dealer_ranges, dealer_sum)
        player_index = self.get_feature(SarsaApproximator.player_ranges, player_sum)

        features = np.zeros((SarsaApproximator.dealer_ranges.shape[0], SarsaApproximator.player_ranges.shape[0], 2))
        for i in dealer_index:
            for j in player_index:
                features[i][j][action] = 1

        return features.flatten()

    def get_action_value(self, dealer_sum, player_sum, action):
        feature = self.featureize(dealer_sum, player_sum, action)
        value = self.action_value_function(feature)
        return value

    def play(self):
        game = State()
        status = STATUS.CONTINUE
        visits = []
        rewards = []
        while status == STATUS.CONTINUE:
            dealer_sum, player_sum = game.dealer_sum(), game.player_sum()

            hit_value = self.get_action_value(dealer_sum, player_sum, 0)
            stick_value = self.get_action_value(dealer_sum, player_sum, 1)

            action = self.greedy_policy(hit_value, stick_value)

            pos = 0 if action==ACTION.HIT else 1
            visits.append((game.dealer_sum(), game.player_sum(), pos))
            status, reward = game.step(action)
            rewards.append(reward)

        steps = len(visits)
        for first in range(steps):
            gt = rt = 0
            weight = 1 - self.lmbda

            first_dealer_sum, first_player_sum, first_action = visits[first]
            for index in range(first, steps):
                reward = rewards[index]

                rt += reward
                if index != 0:
                    weight *= self.lmbda
                if index == steps-1 :
                    weight = weight / (1 - self.lmbda) if self.lmbda!=1 else 1
                    next_q = 0
                else:
                    next_dealer_sum, next_player_sum, next_action = visits[index+1]
                    next_q = self.get_action_value(next_dealer_sum, next_player_sum, next_action)

                gt += weight * (rt + next_q)

            a = 0.01
            features = self.featureize(first_dealer_sum, first_player_sum, first_action)
            delta = gt - self.action_value_function(features)
            self.w += a * delta * features

if __name__ == "__main__":
    Q = np.load("sarsa.npy")

    mse = []
    lmbdas = np.arange(0, 1.01, 0.2)
    for lmbda in lmbdas:
        Q_approx = np.zeros((11, 22, 2))
        total = np.prod(Q_approx[1:, 1:, :].shape)
        m = SarsaApproximator(lmbda=lmbda)
        for i in range(100):
            m.play()
            if i%1000==0:
                print((lmbda, i))

        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    Q_approx[i][j][k] = m.get_action_value(i, j, k)

        value = np.sum(np.square(Q[1:, 1:, :] - Q_approx[1:, 1:, :]))/total
        mse.append(value)
        print(value)

    plt.plot(lmbdas, mse)
    plt.xlabel("lambda")
    plt.ylabel("Mean Squared Error")

    plt.show()

    mse = []
    Q_approx = np.zeros((11, 22, 2))
    total = np.prod(Q_approx[1:, 1:, :].shape)
    m = SarsaApproximator(lmbda=0)
    episodes = 100
    for i in range(episodes):
        m.play()
        if (i%100)==0:
            print(i)
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    Q_approx[i][j][k] = m.get_action_value(i, j, k)

        error = np.sum(np.square(Q[1:, 1:, :] - Q_approx[1:, 1:, :]))/total
        mse.append(error)

    plt.plot(range(episodes), mse)
    plt.xlabel("episodes")
    plt.ylabel("Mean Squared Error")

    mse = []
    Q_approx = np.zeros((11, 22, 2))
    total = np.prod(Q_approx[1:, 1:, :].shape)
    m = SarsaApproximator(lmbda=1.0)
    episodes = 100
    for i in range(episodes):
        m.play()
        if (i%100)==0:
            print(i)
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    Q_approx[i][j][k] = m.get_action_value(i, j, k)

        error = np.sum(np.square(Q[1:, 1:, :] - Q_approx[1:, 1:, :]))/total
        mse.append(error)

    plt.plot(range(episodes), mse)
    plt.xlabel("episodes")
    plt.ylabel("Mean Squared Error")

    pass