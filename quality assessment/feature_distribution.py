"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: feature_distribution.py
 @Function: 绘制各个特征在不同类型信号中的分布
 @DateTime: 2023/4/11 14:29
 @SoftWare: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\result11.8/normalization/10s/"
    filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num_new.txt", encoding='utf-8')
    feature_names = ['BSQI1', 'BSQI2', 'ISQI1', 'ISQI2', 'RsdSQI', 'ASQI1', 'ASQI2', 'PSQI', 'CSQI', 'Avg_corr',
                     'Rsd_corr', 'SNR', 'SSQI', 'KSQI', 'artiafct', 'Sampen', 'MSQI', 'PURSQI', 'AJH', 'AJL']
    # filenames = [584]
    for filename in filenames:
        print("正在处理%s" % filename + '.txt')
        file_num = int(filename)
        filename = str(int(filename))
        feature_data = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
        # feature_data = np.loadtxt(r"E:\研一\信号质量评估\result4.9\PURSQI\955.txt", encoding='utf-8')
        # fig = plt.figure()
        # plt.title(label=filename)
        for j in range(1, len(feature_names) + 1):  # 选择第几个特征
            feature_A, feature_B, feature_C = [], [], []
            for i in range(0, len(feature_data)):
                if feature_data[i, 21] == 1:
                    feature_A.append(feature_data[i, j])
                elif feature_data[i, 21] == 2:
                    feature_B.append(feature_data[i, j])
                elif feature_data[i, 21] == 3:
                    if feature_data[i, 15] == 0:
                        feature_C.append(feature_data[i, j])
            xA = np.arange(0, len(feature_A))
            xB = np.arange(0, len(feature_B))
            xC = np.arange(0, len(feature_C))

            ax = plt.subplot(5, 4, j)
            ax.set_title(label=feature_names[j - 1])
            plt.scatter(xA, feature_A, c='g', alpha=0.1)
            plt.scatter(xB, feature_B, c='b', alpha=0.1)
            plt.scatter(xC, feature_C, c='r', alpha=0.1)

        plt.show()
