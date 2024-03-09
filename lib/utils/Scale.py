from lib.utils.Structs import PhaseData
import numpy as np


class Scale:
    def __init__(self, state_name=None, control_name=None, independence='time', length=1, velocity=1, mass=1):
        """
        归一化单位
        :param length: 长度
        :param velocity: 速度
        :param mass: 质量
        :param state_name: 状态变量的名字
        :param control_name: 控制变量的名字
        """
        self.length = length  # length unit
        self.velocity = velocity  # velocity
        self.mass = mass  # mass
        self.time = length / velocity  # time
        self.accel = self.velocity / self.time  # acceleration
        self.force = self.mass * self.accel  # force
        self.angle = 1.
        self.non = 1.
        self.rho = self.mass / self.length ** 3  # density
        self.angleVel = self.angle / self.time
        self.rate = 1. / self.time

        self.stateScale = np.array([getattr(self, x) for x in state_name]).reshape((-1, 1))
        self.controlScale = np.array([getattr(self, u) for u in control_name]).reshape((-1, 1))
        self.sigmaScale = self.time

    def set_scale(self, state, control, sigma):
        self.stateScale = state.reshape((-1, 1))
        self.controlScale = control.reshape((-1, 1))
        self.sigmaScale = sigma

    def get_scaled(self, phase):
        phase_data = PhaseData()
        phase_data.state = phase.state_bound / self.stateScale
        phase_data.control = phase.control_bound / self.controlScale
        phase_data.sigma = phase.sigma / self.sigmaScale
        return phase_data

    def get_original(self, phase):
        phase_data = PhaseData()
        phase_data.state = phase.state_bound * self.stateScale
        phase_data.control = phase.control_bound * self.controlScale
        phase_data.sigma = phase.sigma * self.sigmaScale
        if self.path is not None:
            phase_data.path = phase.pathFunc * self.path
        return phase_data


class ManualScale:
    def __init__(self, state, control, sigma):
        self.state = state.reshape((-1, 1))
        self.control = control.reshape((-1, 1))
        self.sigma = sigma
