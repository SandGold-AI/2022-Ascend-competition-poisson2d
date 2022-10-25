import mindspore as ms
import numpy as np


class Trainset_poisson():
    def __init__(self, *args):
        self.args = args

    def __call__(self):
        return self.data()

    def data(self):
        n_x = self.args[0]
        n_y = self.args[1]
        n_b = self.args[2]

        # 内部点
        x = np.linspace(0, 1, n_x)
        y = np.linspace(0, 1, n_y)
        x, y = np.meshgrid(x, y)
        xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        xy = xy[(xy[:, 1] - xy[:, 0] - 0.5 < 0) * (xy[:, 1] + xy[:, 0] - 1.5 < 0)]

        # 边界点
        y1 = np.linspace(0, 0.5, n_b)
        x1 = 0 * y1
        xy_b1 = np.concatenate([x1.reshape(-1, 1), y1.reshape(-1, 1)], axis=1)

        x2 = np.linspace(0, 0.5, n_b)
        y2 = x2 + 0.5
        xy_b2 = np.concatenate([x2.reshape(-1, 1), y2.reshape(-1, 1)], axis=1)

        x3 = np.linspace(0.5, 1, n_b)
        y3 = -x3 + 1.5
        xy_b3 = np.concatenate([x3.reshape(-1, 1), y3.reshape(-1, 1)], axis=1)

        y4 = np.linspace(0, 0.5, n_b)
        x4 = 0 * y4 + 1
        xy_b4 = np.concatenate([x4.reshape(-1, 1), y4.reshape(-1, 1)], axis=1)

        x5 = np.linspace(0, 1, n_b)
        y5 = 0 * x5
        xy_b5 = np.concatenate([x5.reshape(-1, 1), y5.reshape(-1, 1)], axis=1)

        xy_b = np.vstack([xy_b1, xy_b2, xy_b3, xy_b4, xy_b5])
        u_b = np.sin(4*np.pi * xy_b[:, [0]]) * np.sin(4*np.pi * xy_b[:, [1]]) / (2 * np.pi ** 2)

        xy = ms.Tensor(xy, dtype=ms.float32)
        xy_b = ms.Tensor(xy_b, dtype=ms.float32)
        u_b = ms.Tensor(u_b, dtype=ms.float32)
        return xy, xy_b, u_b

