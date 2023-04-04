"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: signal_quality_assessment.py
 @Function: Signal feature extraction
 @DateTime: 2022/10/14
 @SoftWare: PyCharm
"""
import numpy as np
import pandas as pd
import os
import Data_utils as DAU
import matplotlib.pyplot as plt
import shutil
import random
import math
import torch
from sampen import sampen2
from scipy import signal


def data(data_root):
    """
    读取数据并转化为数组
    :param data_root:
    :param saveroot:
    :return:
    """
    BCG_sync_, Jpeaks_sync_, Jpeak_DL_, Rpeaks_sync_ = DAU.read_data(data_root)

    BCG = BCG_sync_
    Jpeak_AI = Jpeaks_sync_
    Jpeak_DL = Jpeak_DL_
    Rpeaks = Rpeaks_sync_

    return BCG, Jpeak_AI, Jpeak_DL, Rpeaks


def Mostnum(data):
    """
    :param data: 输入数组
    :return: 返回该数组的众数
    """
    # print("data",data)
    count = {'nan': 0}
    for num in data:
        if np.isnan(num):
            count['nan'] += 1
        else:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
    # print("count",count)
    return {v: k for k, v in count.items()}[max(count.values())]


def distEuclidean(veca, vecb):
    """
    计算欧几里得距离
    """
    return np.sqrt(np.sum(np.square(veca - vecb)))


def MODEL(bcg, ModelLength, jpeaks):
    """
    函数说明：根据该段信号的J峰对应的bcg片段，求平均形成模板
    :param bcg:                     输入待检测模板信号
    :param ModelLength:              输入模板长度
    :param jpeaks:                    输入预设J峰值
    :return:                         返还模板信号
    """
    test = []
    for peak in jpeaks:
        if peak < ModelLength / 2 or (peak + ModelLength / 2) > len(bcg):
            continue
        else:
            test.append(bcg[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))])
    meanBCG = np.zeros(ModelLength)  # 对初始预判J峰的信号段相加平均
    for num in range(len(test)):
        meanBCG += test[num]
    meanBCG = meanBCG / len(test)
    dit = np.array([])  # 计算初始预判信号与平均信号的相似性
    for num in range(len(test)):
        dit = np.append(dit, distEuclidean(test[num], meanBCG) * 1)

    indexmin = np.array([])  # 选择与平均信号最相似的2个原始信号
    for num in range(7):
        if len(dit) > 1:
            indexmin = np.append(indexmin, np.argmin(dit))
            dit[np.argmin(dit)] = float("inf")
        else:
            pass
    indexmin = indexmin.astype(int)
    Model = np.zeros(ModelLength)

    for num in indexmin:
        Model += test[num]
    Model = Model / 7

    return Model


def model_new(bcg, ModelLength, jpeaks):
    """
    参考文献26中的模板构建方法
    :param bcg:
    :param ModelLength:
    :param jpeaks:
    :return:
    """
    model_jpeaks = []
    jpeaks = J_efffective(jpeaks, bcg)
    j_len = len(jpeaks)
    arr = [[0] * j_len for _ in range(j_len)]
    for i in range(0, j_len):
        for j in range(0, j_len):
            ij_rela = np.corrcoef(bcg[int(jpeaks[i] - (ModelLength / 2)):int(jpeaks[i] + (ModelLength / 2))],
                                  bcg[int(jpeaks[j] - (ModelLength / 2)):int(jpeaks[j] + (ModelLength / 2))])
            arr[i][j] = abs(ij_rela[0, 1])
    arr_col = np.mean(arr, axis=0)
    print(arr_col)
    if len(jpeaks):
        for k in range(0, len(arr_col) - 1):
            if arr_col[k] > 0.75 and arr_col[k + 1] > 0.75:
                model_jpeaks.append(jpeaks[k])
    model = np.zeros(ModelLength)
    if len(jpeaks):
        for peaks in model_jpeaks:
            model = bcg[int(peaks - (ModelLength / 2)):int(peaks + (ModelLength / 2))] + model
    Model = model / len(model_jpeaks)

    return arr, Model


def artifact_del(artifact_data, jpeaks1, jpeaks2):
    """
    根据体动文件去除处于体动区间的J峰坐标
    :param artifact_data: 体动数据文件
    :param jpeaks: J峰文件
    :return: 去除体动区间内的J峰后剩余的J峰集合
    """
    for i in range(0, int(len(artifact_data) / 4)):
        min = jpeaks1 > artifact_data[4 * i + 2]
        max = jpeaks1 < artifact_data[4 * i + 3]
        rst = min & max
        j_del = np.where(rst == True)  # 寻找错判J峰
        jpeaks1 = np.delete(jpeaks1, j_del, axis=0)

        min_r = jpeaks2 > artifact_data[4 * i + 2]
        max_r = jpeaks2 < artifact_data[4 * i + 3]
        rst_r = min_r & max_r
        r_del = np.where(rst_r == True)  # 寻找错判J峰
        jpeaks2 = np.delete(jpeaks2, r_del, axis=0)

    jpeaks1_new = jpeaks1.astype(int)
    jpeaks2_new = jpeaks2.astype(int)

    return jpeaks1_new, jpeaks2_new


def SNR(bcg, jpeaks, Model, ModelLength):
    """
    计算信噪比
    :param bcg: BCG信号
    :param jpeaks: J峰
    :param Model: 模板信号
    :param ModelLength: 模板长度
    :return:
    """
    Ps = 0
    Pn = np.linalg.norm(Model, ord=None)  # 计算二范数
    for peak in jpeaks:
        if peak < ModelLength / 2 or (peak + ModelLength / 2) > len(bcg):  # 剔除距离两端小于一半模板长度的J峰点
            continue
        else:
            Ps += np.var(bcg[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))] - Model)
    if len(jpeaks):
        snr = 10 * np.log10(len(jpeaks) * Pn / Ps)
    else:
        snr = 0

    return snr


def J_efffective(jpeaks, bcg):
    """
    去除该片段中前后两端不符合条件的J峰
    :param jpeaks:
    :param bcg:
    :return:
    """
    jpeaks_new = []
    for peak in jpeaks:
        if peak < ModelLength / 2 or (peak + ModelLength / 2) > len(bcg):
            continue
        else:
            jpeaks_new.append(peak)

    return jpeaks_new


def PSQI(bcg, jpeaks, Model, ModelLength):
    """
    计算psqi值，用于衡量各个周期的形态的稳定性
    :param bcg:
    :param jpeaks:
    :param Model:
    :param ModelLength:
    :return:
    """
    relation = []
    j_len = len(jpeaks)
    t = []
    jpeaks = J_efffective(jpeaks, bcg)
    for peak in jpeaks:
        rela = np.corrcoef(bcg[int(peak - (ModelLength / 2)):int(peak + (ModelLength / 2))],
                           Model)  # 计算BCG周期与模板信号的相关系数
        relation.append(rela[0, 1])
        t.append(peak)
    j_len = len(jpeaks)
    arr = [[0] * j_len for _ in range(j_len)]
    for i in range(0, j_len):
        for j in range(0, j_len):
            if i == j:
                arr[i][j] = 0
            else:
                ij_rela = np.corrcoef(bcg[int(jpeaks[i] - (ModelLength / 2)):int(jpeaks[i] + (ModelLength / 2))],
                                      bcg[int(jpeaks[j] - (ModelLength / 2)):int(jpeaks[j] + (ModelLength / 2))])
                arr[i][j] = abs(ij_rela[0, 1])
    avg_corr = np.sum(arr) / ((j_len - 1) * j_len)
    rsd_corr = np.std(arr)/avg_corr
    # print("相关系数矩阵为：", arr)
    if len(relation):
        psqi = sum(i > 0.88 for i in relation) / len(relation)  # 统计高度相关片段的占比
        csqi = sum(i > 0.88 for i in relation) * ModelLength / len(bcg)
    else:
        psqi = 0
        csqi = 0
        avg_corr = 0
        rsd_corr = 1

    # print('csqi:', csqi)

    return psqi, csqi, avg_corr, rsd_corr


def J_match(peaks1, peaks2):
    """
    寻找一定范围内匹配的J点，用于计算BSQI1
    :param peaks1:
    :param peaks2:
    :return:
    """
    len1 = len(peaks1)
    peaks1_match = []
    peaks2_match = []
    match_num = 0
    for m in range(len1):
        A = abs(peaks2 - peaks1[m])
        for n in range(len(A)):
            if A[n] < 50:
                peaks1_match.append(peaks1[m])
                peaks2_match.append(peaks2[n])
                match_num = match_num + sum(A < 50)
                break  # 最多只存在一个与该J峰匹配的点，若已找到，直接跳出此次循环
            else:
                pass

    return peaks1_match, peaks2_match, match_num


def JJ_match(J_peak, new_true_label, bcg_length):
    """
    统计有效的JJ间期，用于计算BSQI2
    :param J_peak:
    :param new_true_label:
    :param bcg_length:
    :return:
    """
    # ----------------构造JJI---------------#
    J_tm_match = []
    J_count = 0
    JJI = np.full(bcg_length, np.nan)
    for num in range(len(J_peak) - 1):
        if 375 < (J_peak[num + 1] - J_peak[num]) < 2000:
            J_tm_match.append(J_peak[num + 1])
            JJI[J_peak[num]: J_peak[num + 1]] = J_peak[num + 1] - J_peak[num]
            J_count = J_count + 1
        else:
            pass
    # ----------------构造RRI---------------#用真实的J峰坐标来计算RRI
    R_count = 0
    J_dl_match = []
    RRI = np.full(bcg_length, np.nan)
    for num in range(len(new_true_label) - 1):
        if 375 < (new_true_label[num + 1] - new_true_label[num]) < 2000:
            J_dl_match.append(new_true_label[num + 1])
            RRI[new_true_label[num]: new_true_label[num + 1]] = new_true_label[num + 1] - new_true_label[num]
            R_count = R_count + 1
        else:
            pass
    # print("间期计算完成，准备进行结果统计")
    find_J_num = J_count + 1
    find_R_num = R_count + 1
    JJI_set = np.array([])
    RRI_set = np.array([])
    for i in range(len(RRI)):  ##将RRI无效的对应的JJI片段去除
        if np.isnan(RRI[i]):
            JJI[i] = np.nan
    # print("RRI对应JJI无效片段删除完成")
    for index in range(len(J_peak) - 1):
        # print("当前正在提取第",index,"个JJI片段")
        stick_J = JJI[J_peak[index]: J_peak[index + 1]]  ##单个JJI片段
        stick_R = RRI[J_peak[index]: J_peak[index + 1]]  ##单个JJI片段
        sig_JJI = Mostnum(stick_J)  ## 返回当个JJI
        sig_RRI = Mostnum(stick_R)  ## 返回当个RRI
        if sig_JJI == 'nan' or sig_RRI == 'nan':
            continue
        else:
            JJI_set = np.append(JJI_set, sig_JJI)
            RRI_set = np.append(RRI_set, sig_RRI)

    return JJI_set, RRI_set


def BSQI(jpeaks1, jpeaks2):
    """
    计算定位算法匹配度(J峰或JJ间期）
    :param jpeaks1:
    :param jpeaks2:
    :return:
    """
    bsqi1 = 0
    bsqi2 = 0
    if len(jpeaks1) and len(jpeaks2):
        peaks1_match, peaks2_match, sum_match = J_match(jpeaks1, jpeaks2)
        bsqi1 = sum_match / (len(jpeaks1) + len(jpeaks2) - sum_match)
        jj1, jj2 = JJ_match(jpeaks1, jpeaks2, scale * fs)  # 统计有效JJ间期
        if len(jj1) and len(jj2):
            jj_sum_match = sum(np.abs(jj1 - jj2) < 50)  # 允许正负30ms的误差范围
            bsqi2 = jj_sum_match / (len(jpeaks1) + len(jpeaks2) - 2 - jj_sum_match)  # JJ间期对得上的占原本定位间期数的比例

    return bsqi1, bsqi2


def sampEn(bcg, std: float, m: int = 2, r: float = 0.15):
    """
    计算时间序列的样本熵
    Input:
        L: 时间序列
        std: 原始序列的标准差
        m: 1或2
        r: 阈值
    Output:
        SampEn
    """
    N = len(bcg)
    B = 10.0
    A = 1.0

    # Split time series and save all templates of length m
    xmi = np.array([bcg[i:i + m] for i in range(N - m)])
    xmj = np.array([bcg[i:i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r * std) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([bcg[i:i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r * std) - 1 for xmi in xm])

    return -np.log(A / B)


def ASQI(peaks, bcg):
    """

    :param peaks:
    :param bcg:
    :return:
    """
    # asqi2= []
    if peaks < ModelLength / 2 or (peaks + ModelLength / 2) > len(bcg):
        asqi2 = 1
    else:
        vall1 = np.argmin(bcg[peaks - 150: peaks]) + peaks - 150  # 寻找左边谷值
        peaks_left = np.argmax(bcg[vall1 - 150: vall1]) + vall1 - 150
        # asqi2.append(peaks_left)
        vall2 = np.argmin(bcg[peaks: peaks + 150]) + peaks  # 寻找右边谷值
        peaks_right = np.argmax(bcg[vall2: vall2 + 150]) + vall2
        # asqi2.append(peaks_right)
        if abs(bcg[peaks]-bcg[peaks_left]) < abs(bcg[peaks]-bcg[peaks_right]):
            asqi2 = abs(bcg[peaks]-bcg[peaks_left])
        else:
            asqi2 = abs(bcg[peaks]-bcg[peaks_right])
        asqi2 = asqi2/bcg[peaks]

    return asqi2


def ISAI(jpeaks, bcg):
    """
    统计JJ间期的稳定性
    :param jpeaks:
    :return:isqi1：JJ间期标准差   isqi2：JJ15/JJ85，
    """
    if len(jpeaks):
        asqi1 = np.std(bcg[jpeaks], ddof=1) / np.mean(bcg[jpeaks])  # J峰幅值的相对标准偏差
        asqi2 = []
        # print("J峰幅度的相对标准偏差为：", asqi)
        for i in range(0, len(jpeaks)):
            asqi = ASQI(jpeaks[i], bcg)
            asqi2.append(asqi)
            np.delete(jpeaks, [i]) if bcg[jpeaks[i]] > 1.2 * np.mean(bcg[jpeaks]) else jpeaks
        # asqi2 = np.std(asqi2)/np.mean(asqi2)
        asqi2 = np.min(asqi2)
        # print(asqi2)
        # if len(asqi2):
        #     plt.figure()
        #     plt.plot(bcg)
        #     plt.plot(jpeaks, bcg[jpeaks], 'x')
        #     plt.plot(asqi2, bcg[asqi2], 'o')
        #     plt.show()
        JJ = np.diff(jpeaks)  # 计算相邻J峰对应的JJ间期
        JJ.sort()
        if len(JJ):
            isqi1 = np.std(JJ, ddof=1)  # 计算JJ间期的标准差
            isqi2 = JJ[int(0.15 * len(JJ))] / JJ[int(0.85 * len(JJ))]
            rsdsqi = isqi1 / np.mean(JJ)  # 间期标准偏差
        else:
            isqi1 = 0
            isqi2 = 0
            rsdsqi = 1
    else:
        isqi1 = 0
        isqi2 = 0
        rsdsqi = 1
        asqi1 = 1
        asqi2 = 0

    isqi1 = 0 if pd.isna((isqi1)) else isqi1
    isqi2 = 0 if pd.isna((isqi2)) else isqi2

    return isqi1, isqi2, rsdsqi, asqi1, asqi2


def picture_show(BCG, Jpeaks1, Jpeaks2, grade, model, scale, i):
    """
    绘图
    :param BCG:
    :param Jpeaks1:
    :param Jpeaks2:
    :param grade:
    :param model:
    :param scale:
    :param i:
    :return:
    """
    plt.figure()
    ax1 = plt.subplot(211)
    plt.title("NO.%d      Quality:%s" % ((i + 1), grade))  # 当前显示片段对应的编号和质量等级
    grade = grade.split("\n")[0]
    if scale == 10:
        if grade == 'a':
            ax1.plot(BCG, linestyle='-', label='BCG')
        elif grade == 'b':
            ax1.plot(BCG, linestyle='-', label='BCG', color='g')
        else:
            ax1.plot(BCG, linestyle='-', label='BCG', color='black')
    else:
        if grade == 'a1':
            ax1.plot(BCG, linestyle='-', label='BCG')
        elif grade == 'a2':
            ax1.plot(BCG, linestyle='-', label='BCG', color='blue')
        elif grade == 'b1':
            ax1.plot(BCG, linestyle='-', label='BCG', color='g')
        elif grade == 'b2':
            ax1.plot(BCG, linestyle='-', label='BCG', color='yellow')
        else:
            ax1.plot(BCG, linestyle='-', label='BCG', color='black')
    ax1.plot(Jpeaks1, BCG[Jpeaks1], 'o', label='Jpeaks')
    ax1.plot(Jpeaks2, BCG[Jpeaks2], 'x', label='Jpeaks')
    ax1.legend(['BCG', 'Jpeaks_DL', 'Jpeaks_TM'])  # 给曲线都上图例

    ax2 = plt.subplot(212)
    plt.title("BCG_model")
    ax2.plot(model, linestyle='-', label='BCG')

    plt.show()


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


def artifact(artifact_data, total_time):
    """
    创建一个与BCG信号等长的信号，体动区域为1，其余为0
    :param artifact_data:
    :param total_time:
    :return:
    """
    atf = np.zeros(total_time)
    for i in range(0, int(len(artifact_data) / 4)):
        atf_start = int(artifact_data[4 * i + 2])
        atf_end = int(artifact_data[4 * i + 3])
        if atf_start > atf_end:  # 若存在此情况，即将两者调换
            print("终止点小于起始点")
            atf_end, atf_start = atf_start, atf_end
        atf[atf_start: atf_end] = 1

    return atf


def Normalization(sqi):
    """
    对特征进行0-1标准化
    :param sqi:
    :return:
    """
    norm_sqi = (sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi))

    return norm_sqi


def SQI(BCG, fs, scale, ModelLength, Jpeak_DL, Jpeak_TM, quality_list, artifact, show_if):
    """
    执行特征提取过程
    :param BCG:
    :param fs: 采样频率
    :param scale: 时间尺度
    :param ModelLength:
    :param quality_file: 质量文件
    :param show_if: 是否绘图
    :return:
    """
    I = []  # 定义空数组，用于存储指标结果
    Bsqi1 = []  # 不同定位算法J峰一致性（允许误差为50ms）
    Bsqi2 = []  # 不同定位算法JJ间期一致性（允许误差为50ms）
    Isqi1 = []  # JJ间期方差
    Isqi2 = []  # 不同排序的JJ间期比值
    Rsdsqi = []  # JJ间期的相对标准偏差
    ASQI1 = []  # J点峰值的相对标准偏差（变异系数）
    ASQI2 = []  # 与J峰峰值最接近的旁边峰值差的相对标准偏差（变异系数）
    Avg_corr = []  # 各个周期间的平均相关系数
    Rsd_corr = []
    Psqi = []  # 相关系数大于0.85的J峰比例
    Csqi = []  # 相关系数大于0.85的时长比例
    Ssqi = []
    Ksqi = []
    Snr = []
    Atf_if = []
    Quality_grade = []
    Samp = []

    for i in range(0, len(quality_list)):  # 信号切片处理
        # print("第", i + 1)
        BCG_cut = BCG[i * scale * fs:(i + 1) * scale * fs - 1]  # 截取第i+1段信号
        # artifact_cut = artifact[i * scale * fs:(i + 1) * scale * fs - 1]  # 截取该片段的体动
        # atf_result = 0
        # if scale == 10:
        #     if sum(artifact_cut == 1):  # 判断该片段是否含有体动，且根据不同时间尺度做不同的处理
        #         atf_result = 1
        # elif scale == 30:
        #     atf_result = sum(artifact_cut == 1) / len(BCG_cut)

        Jpeak_DL_cut = Jpeak_DL[(Jpeak_DL > i * scale * fs + 1) * (
                Jpeak_DL < (i + 1) * scale * fs - 1)] - i * scale * fs  # 该段信号对应的深度学习定位J峰集合
        Jpeak_TM_cut = Jpeak_TM[(Jpeak_TM > i * scale * fs + 1) * (
                Jpeak_TM < (i + 1) * scale * fs - 1)] - i * scale * fs  # 该段信号对应的模板匹配定位J峰集合
        model_DL = MODEL(BCG_cut, ModelLength, Jpeak_DL_cut)  # 深度学习定位结果对应的模板BCG信号
        #
        grade_num = GRADE(quality_list[i].split('\n')[0], scale)  # 将对应等级转换为数字，便于处理
        # # 求取各指标值
        # bsqi1, bsqi2 = BSQI(Jpeak_TM_cut, Jpeak_DL_cut)
        isqi1, isqi2, rsdsqi, asqi1, asqi2 = ISAI(Jpeak_DL_cut, BCG_cut)
        # psqi, csqi, avg_corr, rsd_corr = PSQI(BCG_cut, Jpeak_DL_cut, model_DL, ModelLength)
        # snr = SNR(BCG_cut, Jpeak_DL_cut, model_DL, ModelLength)
        # ssqi = pd.Series(BCG_cut).skew()  # 偏度
        # ksqi = pd.Series(BCG_cut).kurt()  # 峰度
        # BCG_resample = BCG_cut[::10]
        # plt.figure()
        # plt.plot(BCG_cut)
        # plt.plot(BCG_resample)
        # plt.show()
        # samp = sampEn(BCG_resample, np.std(BCG_resample))
        # print("样本熵为：", samp)
        # print(
        #     "I=%f, Bsqi1=%f, Bsqi2=%f, Isqi1=%f, Isqi2=%f, Rsdsqi=%f, ASQI1=%f, ASQI2=%f, Psqi=%f, Csqi=%f, Avg_corr=%f, Rsd_corr=%f, Snr=%f, Ssqi=%f, Ksqi=%f, atf_if=%f, Quality_grade=%f"
        #     % (i, bsqi1, bsqi2, isqi1, isqi2, rsdsqi, asqi1, asqi2, psqi, csqi, avg_corr, rsd_corr, snr, ssqi, ksqi, atf_result, grade_num))
        # # 添加结果至对应的集合中
        I.append(i + 1)
        # Bsqi1.append(bsqi1)
        # Bsqi2.append(bsqi2)
        # Isqi1.append(isqi1)
        # Isqi2.append(isqi2)
        # Rsdsqi.append(rsdsqi)
        # ASQI1.append(asqi1)
        ASQI2.append(asqi2)
        # Psqi.append(psqi)
        # Csqi.append(csqi)
        # Avg_corr.append(avg_corr)
        # Rsd_corr.append(rsd_corr)
        # Snr.append(snr)
        # Ssqi.append(ssqi)
        # Ksqi.append(ksqi)
        # Atf_if.append(atf_result)
        # Samp.append(samp)
        Quality_grade.append(grade_num)
        # 是否绘制图像
        if show_if:
            picture_show(BCG_cut, Jpeak_TM_cut, Jpeak_DL_cut, quality_list[i], model_DL, scale, i)

    return I, ASQI2, Quality_grade

    # return I, Bsqi1, Bsqi2, Isqi1, Isqi2, Rsdsqi, ASQI1, ASQI2, Psqi, Csqi, Avg_corr, Rsd_corr, Snr, Ssqi, Ksqi, Atf_if, Quality_grade


if __name__ == '__main__':
    # --------------------------------------参数设置--------------------------------------
    fs = 1000
    scale = 10  # 信号质量评估的时间尺度
    ModelLength = 850  # 根据统计NN间期均值987、标准差120而定
    show_if = 0
    saveroot = r"E:\研一\信号质量评估\result4.1\ASQI2\30s/"
    data_root = r"E:\BCG_data"
    TM_file = r"E:\BCG_data\备注\模板匹配结果/"
    filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')

    # filenames = [584]
    for filename in filenames:
        print("--------------------------------------正在处理%s.txt--------------------------------------" % filename)
        filename = str(int(filename))
        quality_file = open(os.path.join(data_root, filename + '/' + '质量' + '/' + 'SQ_label1_30s.txt'), 'r+',
                            encoding='utf-8')
        artifact_data = np.loadtxt(os.path.join(data_root, filename + '/' + '体动' + '/' + 'Artifact_a.txt'),
                                   encoding='utf=8')
        Jpeak_TM = np.loadtxt(os.path.join(TM_file, filename + '.txt'), encoding='UTF-8')

        # -----------------------------------------------------------------------------------
        BCG, Jpeak_AI, Jpeak_DL, Rpeaks = data(os.path.join(data_root, filename + '/' + 'Align/'))
        print("信号总时长为：", len(BCG))
        quality_list = quality_file.readlines()  # 读取所有行的元素并返回一个列表
        print("质量片段总数为：", len(quality_list))
        artifact_if = artifact(artifact_data, len(BCG))  # 将体动区间内的点置为1
        print("体动总时长为：", sum(artifact_if == 1))
        # 去除体动区域内的J峰
        Jpeak_DL, Jpeak_TM = artifact_del(artifact_data, Jpeak_DL, Jpeak_TM)

        result = []
        # I, Bsqi1, Bsqi2, Isqi1, Isqi2, Rsdsqi, ASQI1, ASQI2, Psqi, Csqi, Avg_corr, Rsd_corr, Snr, Snr, Ssqi, Ksqi, Atf_if, Quality_grade = \
        #                                                                                               SQI(BCG,
        #                                                                                                   fs,
        #                                                                                                   scale,
        #                                                                                                   ModelLength,
        #                                                                                                   Jpeak_DL,
        #                                                                                                   Jpeak_TM,
        #                                                                                                   quality_list,
        #                                                                                                   artifact_if,
        # #                                                                                                   show_if)
        I, ASQI2, Quality_grade = SQI(BCG, fs, scale, ModelLength, Jpeak_DL, Jpeak_TM, quality_list, artifact_if, show_if)
        result = list(zip(I, ASQI2, Quality_grade ))

        with open(os.path.join(saveroot, filename + '.txt'), 'a+', encoding='utf-8') as f:
            np.savetxt(f, result, '%.0f\t%.5f\t%.0f\t', delimiter='\t')

        # result = list(zip(I, Bsqi1, Bsqi2, Isqi1, Isqi2, Rsdsqi, ASQI1, ASQI2, Psqi, Csqi, Avg_corr, Rsd_corrSnr, Ssqi, Ksqi, Atf_if, Quality_grade))
        #
        # with open(os.path.join(saveroot, filename + '.txt'), 'a+', encoding='utf-8') as f:
        #     np.savetxt(f, result, '%.0f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.0f\t', delimiter='\t')

        print("--------------------------------------%s.txt特征提取完成！--------------------------------------" % filename)
