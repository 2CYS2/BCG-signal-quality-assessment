"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: artifact_detect.py
 @Function: 借用模板匹配中体动检测的方法
 @DateTime: 2023/11/26 21:15
 @SoftWare: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
import os


def windows(data, num):
    """
    函数说明： 颂斌的分窗函数，跟贤哥的写法不太一致
    对输入信号data分窗
    :param data:                  输入数据组
    :param num:                   输入规定每个窗的数据量
    :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
    """
    list = num  # 每个窗num个点
    row = math.ceil(len(data) / list)
    returndata = []
    for i in range(row):
        if i == row:
            returndata.append(data[num * i: -1])
        else:
            returndata.append(data[num * i:num * (i + 1)])

    return returndata


def windows_xian(data, num):
    """
    函数说明：贤哥的分窗函数
    对输入信号data分窗
    :param data:                  输入数据组
    :param num:                   输入规定每个窗的数据量
    :return:                      返还分窗后的二维数组 return[i][j]，i为第i个窗,j为第i个窗的第j个点
    """
    list = num  # 每个窗num个点
    row = int(len(data) / list)
    returndata = np.zeros((row, list))
    for i in range(row):
        for j in range(list):
            returndata[i][j] = data[i * num + j]
    return returndata


def Artifacts_win(data):
    """
    颂斌的去体动程序
    :param data:
    :return:
    """
    Max_value_set = []
    state = []
    Sample_org = 1000
    win = windows(data, 2 * Sample_org)
    for i in range(len(win)):
        Max_value = np.max(win[i])
        Max_value_set.append(Max_value)
    len_30s = int(len(data) / 30 * Sample_org)
    len_60s = int(len(data) / 60 * Sample_org)
    len_120s = int(len(data) / 120 * Sample_org)
    len_240s = int(len(data) / 240 * Sample_org)
    i = 0
    while True:
        i_Count = 0
        if i + 120 <= len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i + 30]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:i + 60]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_240s = Max_value_set[i:i + 120]  # 提取240s内的最大值（120个2s的最大值）

        elif i + 120 > len(Max_value_set) and i + 60 <= len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i + 30]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:i + 60]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_240s = Max_value_set[i:]  # 提取240s内的最大值（120个2s的最大值）

        elif i + 60 > len(Max_value_set) and i + 30 <= len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:i + 30]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_240s = Max_value_set[i:]  # 提取240s内的最大值（120个2s的最大值）

        elif i + 30 > len(Max_value_set) and i + 15 <= len(Max_value_set):
            Max_value_30s = Max_value_set[i:i + 15]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_240s = Max_value_set[i:]  # 提取240s内的最大值（120个2s的最大值）

        else:
            Max_value_30s = Max_value_set[i:]  # 提取30s内的最大值（15个2s的最大值）
            Max_value_60s = Max_value_set[i:]  # 提取60s内的最大值（30个2s的最大值）
            Max_value_120s = Max_value_set[i:]  # 提取120s内的最大值（60个2s的最大值）
            Max_value_240s = Max_value_set[i:]  # 提取240s内的最大值（120个2s的最大值）

        Quartile_30s = np.percentile(Max_value_30s, 25)
        Quartile_60s = np.percentile(Max_value_60s, 25)
        Quartile_120s = np.percentile(Max_value_120s, 25)
        Quartile_240s = np.percentile(Max_value_240s, 25)

        Q1_30s = Quartile_30s * 2  # 提取4s内的最大值的四分位数的1.8倍作为上基线
        Q1_60s = Quartile_60s * 2  # 提取6s内的最大值的四分位数的1.8倍作为上基线
        Q1_120s = Quartile_120s * 2  # 提取8s内的最大值值的四分位数的1.8倍作为上基线
        Q1_240s = Quartile_240s * 2  # 提取10s内的最大值的四分位数的1.8倍作为上基线

        down_Q1_30s = Quartile_30s / 2  # 提取4s内的最大值的四分位数的0.5倍作为下基线
        down_Q1_60s = Quartile_60s / 2  # 提取6s内的最大值的四分位数的0.5倍作为下基线
        down_Q1_120s = Quartile_120s / 2  # 提取8s内的最大值值的四分位数的0.5倍作为下基线
        down_Q1_240s = Quartile_240s / 2  # 提取10s内的最大值的四分位数的0.5倍作为下基线
        # print('Q1_30s',Q1_30s)
        # print('Q1_60s', Q1_60s)
        # print('Q1_120s', Q1_120s)
        # print('Q1_300s', Q1_300s)
        if Max_value_set[i] > Q1_30s or Max_value_set[i] < down_Q1_30s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_60s or Max_value_set[i] < down_Q1_60s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_120s or Max_value_set[i] < down_Q1_120s:
            i_Count = i_Count + 1
        if Max_value_set[i] > Q1_240s or Max_value_set[i] < down_Q1_240s:
            i_Count = i_Count + 1

        if i_Count > 2:
            state.append("Movement")
        else:
            state.append("Sleep")

        i = i + 1

        if i > len(Max_value_set) - 1:
            break

    # --------------------若两个体动之间的距离小于6s，则6s内的信号都被判断为体动------------#
    new_state = copy.deepcopy(state)
    Count_index = 0
    count = 0  ##统计当前片段被判断为体动的个数，
    i = 0
    start_matrix = []
    end_matrix = []
    count_matrix = []
    while True:
        # print('i:',i)
        if i > len(new_state) - 1:
            break
        if new_state[i] == "Sleep" and Count_index == 0 and count == 0:
            i = i + 1
            pass
        elif new_state[i] == "Movement" and Count_index == 0 and count == 0:
            Count_index = 1
            start_index = i
            i = i + 1
        elif new_state[i] == "Movement" and Count_index == 1 and count == 0:
            Count_index = 1
            start_index = i
            i = i + 1
        elif new_state[i] == "Sleep" and Count_index == 1:
            count = count + 1
            i = i + 1
        elif new_state[i] == "Movement" and Count_index == 1 and count != 0:
            Count_index = 0
            end_index = i
            start_matrix.append(start_index)
            end_matrix.append(end_index)
            count_matrix.append(count)
            count = 0
            i = i + 1
    for i in range(len(start_matrix)):
        if 0 < count_matrix[i] <= 2:
            list = ["Movement" for x in range(end_matrix[i] - start_matrix[i])]
            new_state[start_matrix[i]:end_matrix[i]] = list
    #
    # res_state = copy.deepcopy(new_state)
    # for i in range(len(new_state) - 1):  # 将体动前后2s的窗口都设置为体动
    #     if new_state[i] == "Movement":
    #         if i == 1:  # 如果第一个窗口就是体动，则只将后一个2s置为体动
    #             res_state[i + 1] = "Movement"
    #         else:
    #             res_state[i - 1] = "Movement"
    #             res_state[i + 1] = "Movement"
    #     else:
    #         pass

    return np.array(new_state)


def Statedetect(data, threshold=0.5):
    """
    函数说明：
    将输入生理信号进行处理，移除大体动以及空床状态，只保留正常睡眠
    :param data:                输入信号数据
    :param threshold:           设置空床门槛
    :return:                    返还剩余的正常睡眠信号
    """
    win = windows_xian(data, 2000)
    SD = np.zeros(win.shape[0])
    state = []

    for i in range(win.shape[0]):
        SD[i] = np.std(np.array(win[i]), ddof=1)
    Median_SD = np.median(SD)
    # print('meanSTD',Median_SD)
    for i in range(len(SD)):
        # print('Std',SD[i])
        if SD[i] < (Median_SD / 12):  ###以前是固定阈值0.5，固定阈值一般也可以
            state.append("Movement")
        else:
            state.append("Sleep")

    return np.array(state)


if __name__ == '__main__':
    filepath = r"E:\BCG_data/"
    bcg = pd.read_csv(os.path.join(filepath, "972/Align/BCG_sync.txt"), encoding='utf-8').to_numpy()
    artifact_file = np.loadtxt(r"E:\BCG_data\972\体动\Artifact_a.txt", encoding='utf-8')
    data_end = len(bcg) // 2000 * 2000
    # data_end = 12000000
    raw_org = bcg[:data_end]
    print(len(raw_org))
    # filter_ecg = org_ECG[:data_end]
    print('时长：', len(raw_org) / 3600000, '小时')

    First_state = Statedetect(raw_org, 0.5)  ###贤哥的体动判定,现在用于处理空床信号
    Second_state = Artifacts_win(raw_org)  ###颂斌体动，不加std
    print(len(First_state))
    print(len(Second_state))

    bcg_artifact = np.full([len(bcg), 1], 0)
    bcg_artifact1 = np.full([len(bcg), 1], 0)
    Final_state = []
    for i in range(len(Second_state)):
        if First_state[i] == "Movement" or Second_state[i] == "Movement":
            Final_state.append(1)
            bcg_artifact[i * 2 * 1000:(i + 1) * 2 * 1000] = bcg[i * 2 * 1000:(i + 1) * 2 * 1000]
        else:
            Final_state.append(0)

    for j in range(int(len(artifact_file)/4)):
        bcg_artifact1[int(artifact_file[j*4+2]):int(artifact_file[j*4+3])] = bcg[int(artifact_file[j*4+2]):int(artifact_file[j*4+3])]

    artifact_diff = bcg_artifact - bcg_artifact1

    plt.figure()
    plt.plot(bcg, 'b')
    plt.plot(bcg_artifact, 'r')
    plt.plot(artifact_diff, 'g')
    plt.show()
