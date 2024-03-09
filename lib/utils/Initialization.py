"""
author: xmaple
date: 2021-10-16
"""
import numpy as np


class Initialization:
    def __init__(self, method='linear'):
        self.method = method

    def get_initial_trajectory(self, guess):
        guess_init_state = (self.init_state_min + self.init_state_max) / 2
        guess_final_state = (self.final_state_min + self.final_state_max) / 2

        nodes_num = len(nodes)
        collp_num = nodes_num - 1
        x = guess_init_state.reshape((-1, 1)) * (1 - nodes.reshape((1, -1))) / 2 + \
            guess_final_state.reshape((-1, 1)) * (1 + nodes.reshape((1, -1))) / 2
        u = np.zeros((self.udim, collp_num))

        u[0, :] = self.control_max[0] / 2
        u[2, :] = self.control_max[2] / 2

        if self.freetime:
            sigma = (self.sigma_max + self.sigma_min) / 2
        else:
            sigma = self.sigma
        return x, u, sigma
