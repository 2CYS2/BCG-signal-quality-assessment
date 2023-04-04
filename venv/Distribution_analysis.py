"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: Distribution_analysis.py
 @Function: feature distribution analysis
 @DateTime: 2022/10/26
 @SoftWare: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Normalization as NM


def show_10(A, B, C, show_box_or_scatter):
    """
    绘图分析特征
    :param A:
    :param B:
    :param C:
    :param show_box_or_scatter:
    :return:
    """
    if show_box_or_scatter:
        # 绘制箱型图
        label = ['BSQI', 'ISQI1', 'ISQI2', 'PSQI', 'TSQI', 'CSQI', 'CorrSQI', 'SNR', 'SSQI', 'KSQI', 'VarSQI', 'artiafct']
        for i in range(1, 13):
            fig = plt.figure()
            plt.title("Fearure Importance Analyse")
            ax1 = fig.add_subplot(131)
            ax1.boxplot(A[:, i], labels=[label[i - 1] + '_A'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(132)
            ax1.boxplot(B[:, i], labels=[label[i - 1] + '_B'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(133)
            ax1.boxplot(C[:, i], labels=[label[i - 1] + '_C'], widths=0.3, patch_artist=True)  # 描点上色
            plt.show()
    else:
        # 绘制散点图
        fig = plt.figure()
        plt.title("Data Distribution Plot")
        x1 = np.linspace(1, 1, len(A))
        x2 = np.linspace(2, 2, len(B))
        x3 = np.linspace(3, 3, len(C))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)

        ax1 = fig.add_subplot(341)
        plt.title("BSQI")
        ax1.scatter(x1, A[:, 1], marker='o')
        ax1.scatter(x2, B[:, 1], marker='x')
        ax1.scatter(x3, C[:, 1])

        ax2 = fig.add_subplot(342)
        plt.title("ISQI1")
        ax2.scatter(x1, A[:, 2], marker='o')
        ax2.scatter(x2, B[:, 2], marker='x')
        ax2.scatter(x3, C[:, 2])

        ax3 = fig.add_subplot(343)
        plt.title("ISQI2")
        ax3.scatter(x1, A[:, 3], marker='o')
        ax3.scatter(x2, B[:, 3], marker='x')
        ax3.scatter(x3, C[:, 3])

        ax4 = fig.add_subplot(344)
        plt.title("PSQI")
        ax4.scatter(x1, A[:, 4], marker='o')
        ax4.scatter(x2, B[:, 4], marker='x')
        ax4.scatter(x3, C[:, 4])

        ax5 = fig.add_subplot(345)
        plt.title("TSQI")
        ax5.scatter(x1, A[:, 5], marker='o')
        ax5.scatter(x2, B[:, 5], marker='x')
        ax5.scatter(x3, C[:, 5])

        ax6 = fig.add_subplot(346)
        plt.title("CSQI")
        ax6.scatter(x1, A[:, 6], marker='o')
        ax6.scatter(x2, B[:, 6], marker='x')
        ax6.scatter(x3, C[:, 6])

        ax7 = fig.add_subplot(347)
        plt.title("CorrSQI")
        ax7.scatter(x1, A[:, 7], marker='o')
        ax7.scatter(x2, B[:, 7], marker='x')
        ax7.scatter(x3, C[:, 7])

        ax8 = fig.add_subplot(348)
        plt.title("snr")
        ax8.scatter(x1, A[:, 8], marker='o')
        ax8.scatter(x2, B[:, 8], marker='x')
        ax8.scatter(x3, C[:, 8])

        ax9 = fig.add_subplot(3, 4, 9)
        plt.title("SSQI")
        ax9.scatter(x1, A[:, 9], marker='o')
        ax9.scatter(x2, B[:, 9], marker='x')
        ax9.scatter(x3, C[:, 9])

        ax10 = fig.add_subplot(3, 4, 10)
        plt.title("KSQI")
        ax10.scatter(x1, A[:, 10], marker='o')
        ax10.scatter(x2, B[:, 10], marker='x')
        ax10.scatter(x3, C[:, 10])

        ax11 = fig.add_subplot(3, 4, 11)
        plt.title("VarSQI")
        ax11.scatter(x1, A[:, 11], marker='o')
        ax11.scatter(x2, B[:, 11], marker='x')
        ax11.scatter(x3, C[:, 11])

        ax12 = fig.add_subplot(3, 4, 12)
        plt.title("artiafct")
        ax12.scatter(x1, A[:, 12], marker='o')
        ax12.scatter(x2, B[:, 12], marker='x')
        ax12.scatter(x3, C[:, 12])

        fig.legend(['A', 'B', 'C'])  # 给曲线都上图例
        plt.show()

    return None


def show_30(A1, A2, B1, B2, C, show_box_or_scatter):
    """
    绘图分析特征
    :param A1:
    :param A2:
    :param B1:
    :param B2:
    :param C:
    :param show_box_or_scatter:
    :return:
    """
    if show_box_or_scatter:
        # 绘制箱型图
        label = ['BSQI', 'ISQI1', 'ISQI2', 'PSQI', 'TSQI', 'CSQI', 'CorrSQI', 'SSNR', 'SSQI', 'KSQI', 'VarSQI', 'artiafct_ratio']
        for i in range(1, 13):
            fig = plt.figure()
            plt.title("Fearure Importance Analyse")
            ax1 = fig.add_subplot(151)
            ax1.boxplot(A1[:, i], labels=[label[i - 1] + '_A1'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(152)
            ax1.boxplot(A2[:, i], labels=[label[i - 1] + '_A2'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(153)
            ax1.boxplot(B1[:, i], labels=[label[i - 1] + '_B1'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(154)
            ax1.boxplot(B2[:, i], labels=[label[i - 1] + '_B2'], widths=0.3, patch_artist=True)  # 描点上色
            ax1 = fig.add_subplot(155)
            ax1.boxplot(C[:, i], labels=[label[i - 1] + '_C'], widths=0.3, patch_artist=True)  # 描点上色
            plt.show()
    else:
        # 绘制散点图
        fig = plt.figure()
        plt.title("Data Distribution Plot")
        x1 = np.linspace(1, 1, len(A1))
        x2 = np.linspace(2, 2, len(A2))
        x3 = np.linspace(3, 3, len(B1))
        x4 = np.linspace(4, 4, len(B2))
        x5 = np.linspace(5, 5, len(C))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)

        ax1 = fig.add_subplot(341)
        plt.title("BSQI")
        ax1.scatter(x1, A1[:, 1], marker='o')
        ax1.scatter(x2, A2[:, 1], marker='o')
        ax1.scatter(x3, B1[:, 1], marker='x')
        ax1.scatter(x4, B2[:, 1], marker='x')
        ax1.scatter(x5, C[:, 1])

        ax2 = fig.add_subplot(342)
        plt.title("ISQI1")
        ax2.scatter(x1, A1[:, 2], marker='o')
        ax2.scatter(x2, A2[:, 2], marker='o')
        ax2.scatter(x3, B1[:, 2], marker='x')
        ax2.scatter(x4, B2[:, 2], marker='x')
        ax2.scatter(x5, C[:, 2])

        ax3 = fig.add_subplot(343)
        plt.title("ISQI2")
        ax3.scatter(x1, A1[:, 3], marker='o')
        ax3.scatter(x2, A2[:, 3], marker='o')
        ax3.scatter(x3, B1[:, 3], marker='x')
        ax3.scatter(x4, B2[:, 3], marker='x')
        ax3.scatter(x5, C[:, 3])

        ax4 = fig.add_subplot(344)
        plt.title("PSQI")
        ax4.scatter(x1, A1[:, 4], marker='o')
        ax4.scatter(x2, A2[:, 4], marker='o')
        ax4.scatter(x3, B1[:, 4], marker='x')
        ax4.scatter(x4, B2[:, 4], marker='x')
        ax4.scatter(x5, C[:, 4])

        ax5 = fig.add_subplot(345)
        plt.title("TSQI")
        ax5.scatter(x1, A1[:, 5], marker='o')
        ax5.scatter(x2, A2[:, 5], marker='o')
        ax5.scatter(x3, B1[:, 5], marker='x')
        ax5.scatter(x4, B2[:, 5], marker='x')
        ax5.scatter(x5, C[:, 5])

        ax6 = fig.add_subplot(346)
        plt.title("CSQI")
        ax6.scatter(x1, A1[:, 6], marker='o')
        ax6.scatter(x2, A2[:, 6], marker='o')
        ax6.scatter(x3, B1[:, 6], marker='x')
        ax6.scatter(x4, B2[:, 6], marker='x')
        ax6.scatter(x5, C[:, 6])

        ax7 = fig.add_subplot(347)
        plt.title("CorrSQI")
        ax7.scatter(x1, A1[:, 7], marker='o')
        ax7.scatter(x2, A2[:, 7], marker='o')
        ax7.scatter(x3, B1[:, 7], marker='x')
        ax7.scatter(x4, B2[:, 7], marker='x')
        ax7.scatter(x5, C[:, 7])

        ax8 = fig.add_subplot(348)
        plt.title("snr")
        ax8.scatter(x1, A1[:, 8], marker='o')
        ax8.scatter(x2, A2[:, 8], marker='o')
        ax8.scatter(x3, B1[:, 8], marker='x')
        ax8.scatter(x4, B2[:, 8], marker='x')
        ax8.scatter(x5, C[:, 8])

        ax9 = fig.add_subplot(349)
        plt.title("SSQI")
        ax9.scatter(x1, A1[:, 9], marker='o')
        ax9.scatter(x2, A2[:, 9], marker='o')
        ax9.scatter(x3, B1[:, 9], marker='x')
        ax9.scatter(x4, B2[:, 9], marker='x')
        ax9.scatter(x5, C[:, 9])

        ax10 = fig.add_subplot(3, 4, 10)
        plt.title("KSQI")
        ax10.scatter(x1, A1[:, 10], marker='o')
        ax10.scatter(x2, A2[:, 10], marker='o')
        ax10.scatter(x3, B1[:, 10], marker='x')
        ax10.scatter(x4, B2[:, 10], marker='x')
        ax10.scatter(x5, C[:, 10])

        ax11 = fig.add_subplot(3, 4, 11)
        plt.title("VarSQI")
        ax11.scatter(x1, A1[:, 11], marker='o')
        ax11.scatter(x2, A2[:, 11], marker='o')
        ax11.scatter(x3, B1[:, 11], marker='x')
        ax11.scatter(x4, B2[:, 11], marker='x')
        ax11.scatter(x5, C[:, 11])

        ax12 = fig.add_subplot(3, 4, 12)
        plt.title("artiafct")
        ax12.scatter(x1, A1[:, 12], marker='o')
        ax12.scatter(x2, A2[:, 12], marker='o')
        ax12.scatter(x3, B1[:, 12], marker='x')
        ax12.scatter(x4, B2[:, 12], marker='x')
        ax12.scatter(x5, C[:, 12])

        fig.legend(['A1', 'A2', 'B1', 'B2', 'C'])  # 给曲线都上图例
        plt.show()

    return None


if __name__ == '__main__':
    chance = 1  # 选择分析所有样本1或单样本0
    show_box_or_scatter = 1  # 选择展示哪种图形，箱型图1或散点图0
    scale = 30  # 选择处理的尺度
    Mean =[]
    if scale == 10:
        if chance:
            # 导入各等级结果文件(多个样本的总文件)
            A = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\A.txt", encoding='UTF-8')
            B = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\B.txt", encoding='UTF-8')
            C = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\C.txt", encoding='UTF-8')
            # Mean.append(A.mean(axis=0))
            # Mean.append(B.mean(axis=0))
            # Mean.append(C.mean(axis=0))
            # np.savetxt("E:\研一\信号质量评估\Result\result_norm_10\mean.txt",np.array(Mean).T,fmt='%.5f', delimiter='    ')
        else:
            # # 导入某个样本特征结果
            filepath = r"E:\研一\信号质量评估\Result_new\result_30/"
            filenames = '220.txt'
            A, B, C = NM.test_10(filepath, filenames)

        show_10(A, B, C, show_box_or_scatter)

    elif scale == 30:
        if chance:
            # 导入各等级结果文件(多个样本的总文件)
            A1 = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\A1.txt", encoding='UTF-8')
            A2 = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\A2.txt", encoding='UTF-8')
            B1 = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\B1.txt", encoding='UTF-8')
            B2 = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\B2.txt", encoding='UTF-8')
            C = np.loadtxt(r"E:\研一\信号质量评估\Result_new\新建文件夹\C_30.txt", encoding='UTF-8')
            # Mean.append(A1.std(axis=0))
            # Mean.append(A2.std(axis=0))
            # Mean.append(B1.std(axis=0))
            # Mean.append(B2.std(axis=0))
            # Mean.append(C.std(axis=0))
            # np.savetxt(r"E:\研一\信号质量评估\Result\result_norm_30\std.txt",np.array(Mean).T,fmt='%.5f', delimiter='    ')
        else:
            # # 导入某个样本特征结果
            filepath = r"E:\研一\信号质量评估\Result_new\result_30/"
            filenames = '220.txt'
            A1, A2, B1, B2, C = NM.test_30(filepath, filenames)

        show_30(A1, A2, B1, B2, C, show_box_or_scatter)
