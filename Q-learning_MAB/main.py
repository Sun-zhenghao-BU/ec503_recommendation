import numpy as np
from numpy import random
from MABenv import MAB
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

e_greedy = 0.7

if __name__ == '__main__':
    mab = MAB(numOfMAB=10)

    for i in range(10000):
        if i == 5000:
            pass
            xx=1

        actionC = random.rand()
        if actionC <= e_greedy:
            action = np.argmax(mab.Q)
        else:
            action = np.random.randint(10,size = 1)
        mab.step(int(action))
        if i%500==0:
            print('Possibility of winning: ')
            print(mab.Q)
            print('times of pulling of each machine: ')
            print(mab.pull)
            print()