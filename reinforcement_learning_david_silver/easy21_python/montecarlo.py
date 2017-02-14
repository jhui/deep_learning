import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from game21 import *
from agent import *

class MonteCarlo(Agent):

    def play(self):
        visits, rewards = super().play()

        gt = 0
        for dealer_sum, player_sum, action in reversed(visits):
            a = 1 / self.N[dealer_sum, player_sum, action]
            current = self.Q[dealer_sum, player_sum, action]
            gt += rewards.pop()
            self.Q[dealer_sum, player_sum, action] += a * (gt - current)

if __name__ == "__main__":
    do_trainning = False
    do_saving = True
    if do_trainning:
        m = MonteCarlo()
        for i in range(10000000):
            m.play()
            if i%200000==0:
                print(i)

        Q = m.Q
        if do_saving:
            np.save("montecarlo.npy", Q)
    else:
        Q = np.load("montecarlo.npy")


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

    pass