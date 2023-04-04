"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: test.py
 @Function: 
 @DateTime: 2022/10/26 15:59
 @SoftWare: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math


def ex(data):
    result = []
    for i in data:
        result.append(math.exp(i))

    return result


def GRADE(grade, scale):
    """
    将对应等级的字符转化为数字
    :param grade:
    :return:
    """
    grade_num = 0
    if scale == 30:
        if grade == 'a1':
            grade_num = 1
        elif grade == 'a2':
            grade_num = 2
        elif grade == 'b1':
            grade_num = 3
        elif grade == 'b2':
            grade_num = 4
        elif grade == 'c':
            grade_num = 5
    elif scale == 10:
        if grade == 'a':
            grade_num = 1
        elif grade == 'b':
            grade_num = 2
        elif grade == 'c':
            grade_num = 3

    return grade_num


if __name__ == '__main__':
    # BCG_old = pd.read_csv(r"E:\BCG_data\1296\old\Align\BCG_sync.txt", encoding='utf-8').to_numpy()
    # BCG = pd.read_csv(r"E:\BCG_data\1296\Align\BCG_sync.txt", encoding='utf-8').to_numpy()
    # artifact_data = np.loadtxt(r"E:\BCG_data\1296\old\体动\Artifact_a.txt", encoding='utf=8')
    #
    # for i in range(0, int(len(artifact_data)/4)):
    #     artifact_data[4 * i + 2] = artifact_data[4 * i + 2] - 73
    #     artifact_data[4 * i + 3] = artifact_data[4 * i + 3] - 73
    #
    # print(artifact_data)

    # print(len(BCG))
    # # J_real = np.loadtxt(r"E:\BCG_data\1296\Align\Jpeaks_sync.txt", encoding='utf-8')
    # # J_DL = np.loadtxt(r"E:\BCG_data\1296\Align\Jpeak_DL.txt", encoding='utf-8')
    #
    # # J_real = J_real.astype('int64')
    # # J_real = J_real + 360
    # # J_real_new = []
    # # for i in range(0, len(J_real)):
    # #     J = BCG[J_real[i]-50 : J_real[i]+50]
    # #     J_new = np.argmax(J)+ J_real[i]-50
    # #     J_real_new.append(J_new)
    #
    # # J_DL = J_DL.astype('int64')
    # fig = plt.figure()
    # plt.plot(BCG_old)
    # # plt.plot(J_real, BCG[J_real], 'o',)
    # plt.plot(BCG + 500)
    # # plt.plot(J_real_new, BCG[J_real_new] + 500, 'o',)
    # np.savetxt(r"E:\BCG_data\1296\体动\Artifact_a.txt", artifact_data, fmt='%.0f')
    # plt.show()
    #
    # 数据文件路径
    filepath = r"E:\研一\信号质量评估\result3.29\10s/"
    sampenpath = r"E:\研一\信号质量评估\result4.1\ASQI2\10s/"
    saveroot = r"E:\研一\信号质量评估\result4.1\10s/"
    # filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')
    # filenames = [1010]  # 获取该文件夹中所有样本的特征结果
    filenames = [551, 584, 726, 549, 582, 671, 955, 1354, 286, 560, 586]
    for filename in filenames:
        filename = str(int(filename))
        file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
        quality_data = np.loadtxt(os.path.join(r'E:\研一\信号质量评估\result4.1\quality_index/', filename + '.txt'), encoding='utf-8')
        sampen = np.loadtxt(os.path.join(sampenpath, filename + '.txt'), encoding='utf-8')
        # new_file = np.insert(file[:,0:15], 15, sampen[:, 1], axis=1)
        # new_file = np.insert(new_file, 16, sampen[:, 2], axis=1)
        # new_file = np.insert(new_file, 17, file[:, 15], axis=1)

        # isqi2_e = ex(file[:, 4])
        # rsdsqi_e = ex(file[:, 5])
        # print(np.shape(isqi2_e))
        # new_file = np.insert(file[:, 0:19], 19, isqi2_e, axis=1)
        # new_file = np.insert(new_file, 18, rsdsqi_e, axis=1)
        # new_file = np.insert(new_file, 20, file[:, 19], axis=1)

        new_file = np.insert(file[:, 0:19], 19, sampen[:, 1], axis=1)
        # new_file = np.insert(new_file, 20, file[:, 19], axis=1)
        new_file = np.insert(new_file, 20, quality_data, axis=1)

        print(np.shape(new_file))
        np.savetxt(os.path.join(saveroot, str(int(filename)) + '.txt'), new_file, fmt='%.6f\t')

    # scale = 10
    # filenames = [551, 584, 726, 549, 582, 671, 955, 1354, 286, 560, 586]
    # for filename in filenames:
    #     quality_index = []
    #     filename = str(int(filename))
    #     quality_data = open(os.path.join(r'E:\BCG_data/', filename + '/' + '质量\old\SQ_label1_10s.txt'), 'r+', encoding='utf-8')
    #     quality_list = quality_data.readlines()  # 读取所有行的元素并返回一个列表
    #     for i in range(0, len(quality_list)):
    #         grade_num = GRADE(quality_list[i].split('\n')[0], scale)  # 将对应等级转换为数字，便于处理
    #         quality_index.append(grade_num)
    #         print(grade_num)
    #
    #     np.savetxt(os.path.join(r'E:\研一\信号质量评估\result4.1\quality_index/', filename + '.txt'), quality_index, fmt='%.0f')

