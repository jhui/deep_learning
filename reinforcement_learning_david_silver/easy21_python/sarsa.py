import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from game21 import *
from agent import *

class Sarsa(Agent):
    def __init__(self, lmbda=0.0, N0=100.0):
        super().__init__(N0=N0)
        self.lmbda = lmbda

    def play(self):
        visits, rewards = super().play()
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
                    next_q = self.Q[next_dealer_sum, next_player_sum, next_action]

                gt += weight * (rt + next_q)

            a = 1 / self.N[first_dealer_sum, first_player_sum, first_action]
            current = self.Q[first_dealer_sum, first_player_sum, first_action]
            self.Q[first_dealer_sum, first_player_sum, first_action] += a * (gt - current)

if __name__ == "__main__":
    do_trainning = False
    do_saving = True
    if do_trainning:
        m = Sarsa(lmbda=0.6)
        for i in range(1000000):
            m.play()
            if i%200000==0:
                print(i)

        Q = m.Q
        if do_saving:
            np.save("sarsa.npy", Q)
    else:
        Q = np.load("sarsa.npy")

    V = np.amax(Q, axis=2)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x = range(1, 11)
    y = range(1, 22)
    X, Y = np.meshgrid(x, y)
    # ax.plot_wireframe(X.T, Y.T, best_q_value[1:, 1:])
    ax.plot_surface(X.T, Y.T, V[1:, 1:], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("Dealer card")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("V")

    fig.show()

    mse = []
    lmbdas = np.arange(0, 1.01, 0.1)
    total = np.prod(Q[1:, 1:, :].shape)
    for lmbda in lmbdas:
        m = Sarsa(lmbda=lmbda)
        for i in range(1000):
            m.play()

        Q_lmbda = m.Q
        mse.append(np.sum(np.square(Q[1:, 1:, :] - Q_lmbda[1:, 1:, :]))/total)

    plt.plot(lmbdas, mse)
    plt.xlabel("lambda")
    plt.ylabel("Mean Squared Error")

    plt.show()

    mse = []
    m = Sarsa(lmbda=0)
    episodes = 10000
    for i in range(episodes):
        m.play()
        Q_lmbda = m.Q
        error = np.sum(np.square(Q[1:, 1:, :] - Q_lmbda[1:, 1:, :]))/total
        mse.append(error)

    plt.plot(range(episodes), mse)
    plt.xlabel("episodes")
    plt.ylabel("Mean Squared Error")

    mse = []
    m = Sarsa(lmbda=1)
    for i in range(episodes):
        m.play()
        Q_lmbda = m.Q
        error = np.sum(np.square(Q[1:, 1:, :] - Q_lmbda[1:, 1:, :]))/total
        mse.append(error)

    plt.plot(range(episodes), mse)
    plt.xlabel("episodes")
    plt.ylabel("Mean Squared Error")

    pass