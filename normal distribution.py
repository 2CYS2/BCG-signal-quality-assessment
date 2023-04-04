"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: normal distribution.py
 @Function: 判断特征值是否符合正态分布
 @DateTime: 2022/10/27 20:53
 @SoftWare: PyCharm
"""
# 导入scipy模块
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # data = np.loadtxt(r"E:\研一\信号质量评估\Result\result\1000.txt", encoding='UTF-8')  # 读取数据
    label = ['BSQI1', 'BSQI2','ISQI1', 'ISQI2', 'RsdSQI', 'ASQI', 'PSQI', 'CSQI', 'Avg_corr', 'SNR', 'SSQI', 'KSQI', 'artiafct']
    filepath = r"E:\研一\信号质量评估\result3.21\10s/"
    filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果

    for filename in filenames:
        data = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
        print(np.shape(data))

        # 去除特征中的异常值（nan和inf）
        nan_con = np.where(np.isnan(data))[0]
        data = np.delete(data, nan_con, axis=0)  # 执行删除 只删除了第2个数据

        inf_con = np.where(np.isinf(data))[0]
        data = np.delete(data, inf_con, axis=0)  # 执行删除 只删除了第2个数据

        for i in range(1,14):
            u = data[:,i].mean()  # 计算均值
            std = data[:,i].std()  # 计算标准差
            print(label[i-1]+"的结果为：")
            result = stats.kstest(data[:, i], 'norm', (u, std))
            print(result)

        # 画散点图和直方图
        fig = plt.figure()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
        ax1 = fig.add_subplot(341)
        plt.title(label[0])
        x0 = range(0, len(data[:,1]))
        plt.scatter(x0, data[:,1])
        # plt.hist(data[:,0],bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax1)

        ax2 = fig.add_subplot(342)
        plt.title(label[1])
        x1 = range(0, len(data[:,2]))
        plt.scatter(x1, data[:,2])
        # plt.hist(data[:, 1], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax2)

        ax3 = fig.add_subplot(343)
        plt.title(label[2])
        x2 = range(0, len(data[:,3]))
        plt.scatter(x2, data[:,3])
        # plt.hist(data[:, 2], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax3)

        ax4 = fig.add_subplot(344)
        plt.title(label[3])
        x3 = range(0, len(data[:,4]))
        plt.scatter(x3, data[:,4])
        # plt.hist(data[:, 3], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax4)

        ax5 = fig.add_subplot(345)
        plt.title(label[4])
        x4 = range(0, len(data[:,5]))
        plt.scatter(x4, data[:,5])
        # plt.hist(data[:, 4], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax5)

        ax6 = fig.add_subplot(346)
        plt.title(label[5])
        x5 = range(0, len(data[:,6]))
        plt.scatter(x5, data[:,6])
        # plt.hist(data[:, 5], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax6)

        ax7 = fig.add_subplot(347)
        plt.title(label[6])
        x6 = range(0, len(data[:,7]))
        plt.scatter(x6, data[:,7])
        # plt.hist(data[:, 6], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax7)

        ax8 = fig.add_subplot(348)
        plt.title(label[7])
        x7 = range(0, len(data[:,8]))
        plt.scatter(x7, data[:,8])
        # plt.hist(data[:,7],bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax8)

        ax9 = fig.add_subplot(349)
        plt.title(label[8])
        x8 = range(0, len(data[:,9]))
        plt.scatter(x8, data[:,9])
        # plt.hist(data[:, 8], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax9)

        ax10 = fig.add_subplot(3, 4, 10)
        plt.title(label[9])
        x9 = range(0, len(data[:,10]))
        plt.scatter(x9, data[:,10])
        # plt.hist(data[:, 9], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax10)

        ax11 = fig.add_subplot(3, 4, 11)
        plt.title(label[10])
        x10 = range(0, len(data[:,11]))
        plt.scatter(x10, data[:,11])
        # plt.hist(data[:, 10], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax11)
        #
        ax12 = fig.add_subplot(3, 4, 12)
        plt.title(label[11])
        x11 = range(0, len(data[:,12]))
        plt.scatter(x10, data[:,12])
        # plt.hist(data[:, 11], bins=30, alpha=0.5)
        # plt.plot(kind='kde', secondary_y=True, ax=ax12)
        plt.show()