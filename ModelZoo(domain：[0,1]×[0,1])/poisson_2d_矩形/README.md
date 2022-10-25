## 问题描述

二维Poisson方程
$$
\Delta u=-\sin (4 \pi x) \sin (4 \pi y),  \, (x,y) \in \Omega \\
u = \frac{1}{2(4\pi)^2}\sin (4 \pi x) \sin (4 \pi y),  \, (x,y) \in \partial\Omega
$$
其中 $\Omega$ 代表求解区域，我们求解了几种常用的几何边界：矩形、圆、三角形、五边形。



## PINNs方法求解Poisson方程

PINNs方法的优化目标为：
$$
\min _\theta \mathcal{L}_{\text {train }}(\theta)=\lambda_r \mathcal{L}_r(\theta)+\lambda_{b c} \mathcal{L}_{b c}(\theta),
$$
其中， $\mathcal{L}_r$ 和 $\mathcal{L}_{bc}$ 分别代表内部残差和边界损失值。$\lambda=\left\{\lambda_r, \lambda_{b c} \right\}$ 是损失函数的权重，能够均衡收敛不同损失项的收敛速度。

其中 $\mathcal{L}_r$ 和 $\mathcal{L}_{bc}$ 可分别表示为：
$$
\mathcal{L}_r(\theta)=\frac{1}{N_r} \sum_{i=1}^{N_r}\left|\mathcal{N}\left[u_{\mathcal{N} \mathcal{N}_\theta}\right]\left(x_r^i\right)-f\left(x_r^i\right)\right|^2\\ \mathcal{L}_{b c}(\theta)=\frac{1}{N_{b c}} \sum_{i=1}^{N_{b c}}\left|\mathcal{B}\left[u_{\mathcal{N} \mathcal{N}_\theta}\right]\left(x_{b c}^i\right)-g\left(x_{b c}^i\right)\right|^2
$$
这里的 $\mathcal{N}[\cdot]$ 和 $\mathcal{B}[\cdot]$ 分别是控制方程的微分算子和边界条件。



## 模型结构

我们采用了Modified MLP网络结构：
$$
\begin{aligned}
&U=\phi\left(X W^1+b^1\right), \quad V=\phi\left(X W^2+b^2\right) \\
&H^{(1)}=\phi\left(X W^{z, 1}+b^{z, 1}\right) \\
&Z^{(k)}=\phi\left(H^{(k)} W^{z, k}+b^{z, k}\right), \quad k=1, \ldots, L \\
&H^{(k+1)}=\left(1-Z^{(k)}\right) \odot U+Z^{(k)} \odot V, k=1, \ldots, L \\
&f_\theta(x)=H^{(L+1)} W+b
\end{aligned}
$$
Modified MLP模型结构图如下：

<img src="/Users/guangtao/Desktop/华为Mindspore比赛/pictures/modified_MLP.png" alt="modified_MLP" style="zoom:50%;" />



该模型结构参考了文章（Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient pathologies in physics-informed neural networks），比起全连接能够更加容易地抓取PDE解中带有剧烈变化的部分。



## 数据集

PINNs方法可以根据物理信息无监督地学习PDE方程，数据集是通过采样得到的。在本算法中，我们的训练数据和测试数据都是根据物理区域直接生成的，具体方式为：对于二维几何建模采样方法，我们采用矩形框住目标多边形区域，在矩形框内等距采样，然后依靠判断条件获取对应几何区域内部的采样点；对于目标多边形的边界，我们计算出直线的方程，在每一条边上都进行等距采样(对于圆形边界，我们在$[0,2\pi]$上对角度进行等距采样后通过变换得到边界点)。



## 运行环境要求

计算硬件：Ascend 计算芯片

计算框架：Mindspore 1.7.0，numpy 1.21.2，matplotlib 3.5.1，scipy 1.5.4



## 代码框架

```
.
└─PINNforPoisson
  ├─README.md
  ├─src
    ├──config.py                      # parameter configuration
    ├──dataset.py                     # dataset
    ├──model.py                       # network structure
    ├──eager_lbfgs.py                 # L-BFGS algorithm
  ├──solve.py                         # train and test
```





## 超参数设置

```
import argparse

class Options_poisson(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--decay_rate', type=float, default=0.8,
                            help='decay_rate in lr_scheduler for Adam optimizer')
        parser.add_argument('--step_per_epoch', type=int, default=2500,
                            help='step size in lr_scheduler for Adam optimizer')
        parser.add_argument('--decay_steps', type=int, default=1, help='衰减的step数')
        parser.add_argument('--epochs_Adam', type=int, default=50000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=5000, help='epochs for LBFGS optimizer')
        parser.add_argument('--dim_hidden', type=int, default=128, help='neurons in hidden layers')
        parser.add_argument('--hidden_layers', type=int, default=6, help='number of hidden layers')
        parser.add_argument('--n_x', type=int, default=101, help='number of interior point samples on the X-axis')
        parser.add_argument('--n_y', type=int, default=101, help='number of interior point samples on the Y-axis')
        parser.add_argument('--n_b', type=int, default=256, help='number of boundary point samples on the every axis')
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        return arg
```



## 模型训练

可以直接使用solve.py文件进行PINNs模型训练和求解Poisson方程。在训练过程中，模型的参数和训练过程也会被自动保存

```
python solve.py
```

模型的损失会实时展示出来，变化如下（以矩形区域为例）：

```
Epoch #  100   loss:4.31e+00   time:74.59
Epoch #  200   loss:2.53e-01   time:2.00
Epoch #  300   loss:5.98e-02   time:2.00
Epoch #  400   loss:1.98e-02   time:2.00
Epoch #  500   loss:1.65e-02   time:2.01
Epoch #  600   loss:1.76e-02   time:2.06
Epoch #  700   loss:1.14e-02   time:2.05
Epoch #  800   loss:8.94e-03   time:2.05
Epoch #  900   loss:3.28e-01   time:2.05
Epoch #  1000   loss:3.41e-03   time:2.04
Epoch #  1100   loss:7.21e-03   time:2.04
Epoch #  1200   loss:3.21e-03   time:2.03
Epoch #  1300   loss:5.60e-03   time:2.05
Epoch #  1400   loss:1.50e-02   time:2.06
...
Epoch #  49600   loss:3.58e-05   time:2.06
Epoch #  49700   loss:3.58e-05   time:2.05
Epoch #  49800   loss:3.58e-05   time:2.05
Epoch #  49900   loss:3.58e-05   time:2.06
Epoch #  50000   loss:3.58e-05   time:2.06
内部点的Error u: 4.844291e-04
```



## MindScience官网

可以访问官网以获取更多信息：https://gitee.com/mindspore/mindscience