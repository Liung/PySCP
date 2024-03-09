import warnings

import numpy as np
from lib.Mesh.MeshPS import LegendreGauss, LegendreGaussRadau, FlippedLegendreGaussRadau

"""
xtao: 离散点
nx: 离散点个数
cps: 配点，
ncp: 配点个数
degrees: 每个分段内的多项式阶数 [数组]
seg_fractions: 每个分段占整个归一化时域[-1, 1]的百分比
seg_legendre: 每个分段的勒让德离散方法，对应一个MeshPS对象
segBoundaryPoints: 每个分段
cp_state, cp_control: 循环配点时，状态变量和控制变量的索引
nx, nu: 状态变量和控制变量个数。
PDM, PIM: 微分与积分矩阵
PIMX0Index, PIMXfIndex: 各个配点在积分形式下的起始点与
weights4State, weights4Control: 离散积分型目标函数

"""


class HpPseudoSpectral:
    def __init__(self, degree, seg_fractions):
        """
         generalized hp pseudo spectral method
        :param degree: list of polynomial degrees in each interval
        :param seg_fractions: ratio between each interval to the whole time domain length
        also known as interval, sub-interval or sub-domain.
        """
        degree = np.array(degree).astype(int)  # convert to ndarray
        self.ncp = np.sum(degree)
        self.nu = self.ncp

        self.nseg = len(degree)  # number of segments
        self.seg_degrees = degree  # degrees in each segment
        self.seg_fractions = seg_fractions  # fraction of each segment

        # 状态变量和控制变量在离散点数组中的索引
        self.cp_state = np.zeros((self.ncp,), dtype=int)
        self.nx = None
        self.xtao = None
        self.utao = None
        self.PDM = None
        self.PIM = None
        self.cps = None
        self.seg_legendre = None
        self.weights4State = None
        self.weights4Control = None
        self.cpControl = np.arange(self.ncp)

    def setup(self, deg1, lg=False):
        cps = []
        self.xtao = np.zeros((self.nx,))
        self.utao = np.zeros((self.nu,))
        self.PDM = np.zeros((self.ncp, self.nx))
        self.PIM = np.zeros((self.nx - 1, self.ncp))

        self.weights4State = np.zeros((self.nseg, self.nx))
        seg_boundary_points = np.concatenate([[0], np.cumsum(self.seg_fractions)]) * 2 - 1
        for i in range(self.nseg):
            tao0 = seg_boundary_points[i]
            taof = seg_boundary_points[i + 1]
            cps.append(self.seg_legendre[i].cps * (taof - tao0) / 2 + (taof + tao0) / 2)
            # segment mesh-points in the whole mesh-points
            segx_in_xs = slice(np.sum(deg1[:i]), np.sum(deg1[:i + 1]) + 1)
            # segment cps in the whole cps
            segcps_in_cps = slice(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1]))
            # segment integrated points in all integrated points
            seg_mps_in_integrated = slice(np.sum(deg1[:i]), np.sum(deg1[:i + 1]))

            self.xtao[segx_in_xs] = self.seg_legendre[i].xtao * (taof - tao0) / 2 + (taof + tao0) / 2
            self.weights4State[i, segx_in_xs] = self.seg_legendre[i].integral_weights * self.seg_fractions[i]
            self.PDM[segcps_in_cps, segx_in_xs] = self.seg_legendre[i].PDM / self.seg_fractions[i]  # PDM divide
            if lg:
                # PIM multiply
                self.PIM[seg_mps_in_integrated, segcps_in_cps] = self.seg_legendre[i].PIM * self.seg_fractions[i]
            else:
                # PIM multiply
                self.PIM[segcps_in_cps, segcps_in_cps] = self.seg_legendre[i].PIM * self.seg_fractions[i]

        self.cps = np.hstack(cps)
        self.utao = self.cps
        self.weights4State = np.squeeze(np.sum(self.weights4State, axis=0))
        self.weights4Control = self.weights4State[self.cp_state]


class HpLG(HpPseudoSpectral):
    def __init__(self, degree, seg_fractions):
        super().__init__(degree=degree, seg_fractions=seg_fractions)
        self.nx = self.ncp + self.nseg + 1
        self.seg_legendre = [LegendreGauss(d) for d in degree]

        deg1 = self.seg_degrees + 1
        left_bound = np.concatenate([[0], np.cumsum(deg1[:-1])])
        self.PIMX0Index = np.hstack([[left_bound[i]] * (self.seg_degrees[i] + 1) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([left_bound[i] + np.arange(1, self.seg_degrees[i] + 2) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1]))  # segment cps in the whole cps
            self.cp_state[seg_cps_in_cps] = np.arange(np.sum(deg1[:i]) + 1, np.sum(deg1[:i + 1]))
        super().setup(deg1, lg=True)


class HpLGR(HpPseudoSpectral):
    def __init__(self, degree, seg_fractions):
        super().__init__(degree=degree, seg_fractions=seg_fractions)
        self.nx = self.ncp + 1
        self.nx = self.nx
        self.seg_legendre = [LegendreGaussRadau(d) for d in degree]

        left_bound = np.concatenate([[0], np.cumsum(self.seg_degrees[:-1])])
        self.PIMX0Index = np.hstack([[left_bound[i]] * (self.seg_degrees[i]) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([left_bound[i] + np.arange(1, self.seg_degrees[i] + 1) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1]))  # segment cps in the whole cps
            self.cp_state[seg_cps_in_cps] = np.arange(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1]))
        super().setup(self.seg_degrees)


class HpfLGR(HpPseudoSpectral):
    def __init__(self, degree, seg_fractions):
        super().__init__(degree=degree, seg_fractions=seg_fractions)
        self.nx = self.ncp + 1
        self.nx = self.nx
        self.seg_legendre = [FlippedLegendreGaussRadau(d) for d in degree]

        left_bound = np.concatenate([[0], np.cumsum(self.seg_degrees[:-1])])
        self.PIMX0Index = np.hstack([[left_bound[i]] * (self.seg_degrees[i]) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([left_bound[i] + np.arange(1, self.seg_degrees[i] + 1) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1]))  # segment cps in the whole cps
            self.cp_state[seg_cps_in_cps] = np.arange(np.sum(self.seg_degrees[:i]), np.sum(self.seg_degrees[:i + 1])) + 1
        super().setup(self.seg_degrees)


if __name__ == '__main__':
    hplg = HpLG([6, 6, 6], seg_fractions=[1 / 3] * 3)
    hplgr = HpLGR([6, 6, 6], seg_fractions=[1 / 3] * 3)
    hpflgr = HpfLGR([6, 6, 6], seg_fractions=[1 / 3] * 3)
    print(hplg.PDM.shape, hplg.PIM.shape, hplg.cp_state)
    print(hplgr.PDM.shape, hplgr.PIM.shape, hplgr.cp_state)
    print(hpflgr.PDM.shape, hpflgr.PIM.shape, hpflgr.cp_state)
    print(np.sum(hplg.weights4State))

    lg = LegendreGauss(6)
    lgr = LegendreGaussRadau(6)
    flgr = FlippedLegendreGaussRadau(6)
    print(lg.PDM.shape, lg.PIM.shape)
    print(lgr.PDM.shape, lgr.PIM.shape)
    print(flgr.PDM.shape, flgr.PIM.shape)
