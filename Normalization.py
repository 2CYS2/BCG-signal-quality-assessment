"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: Normalization.py
 @Function: Data normalization（单个个体标准化后按质量划分为多个文件）
 @DateTime: 2022/10/25
 @SoftWare: PyCharm
"""
# 绘制图像分析特征的统计分布
import numpy as np
import matplotlib.pyplot as plt
import os


def grade_10(features):
    """
    按不同类别进行划分
    :param features:
    :return:
    """
    A = []
    B = []
    C = []
    for i in range(0, len(features)):
        if features[i, 13] == 1:
            A.append(features[i, :])
        elif features[i, 13] == 2:
            B.append(features[i, :])
        elif features[i, 13] == 3:
            C.append(features[i, :])

    return A, B, C


def test_10(filepath, filename):
    """
    去除特征值中的nan和inf等异常值
    :param filepath:
    :param filename:
    :return:
    """
    features = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
    print(np.shape(features))

    # 去除特征中的异常值（nan和inf）
    nan_con = np.where(np.isnan(features))[0]
    print("文件%s的inf位置为：" % filename, nan_con)
    features = np.delete(features, nan_con, axis=0)  # 执行删除 只删除了第2个数据
    # print(np.shape(features))

    inf_con = np.where(np.isinf(features))[0]
    print("文件%s的nan位置为：" % filename, inf_con)
    features = np.delete(features, inf_con, axis=0)  # 执行删除 只删除了第2个数据
    # print(np.shape(features))

    features_new = []
    features_new.append(features[:, 0])
    # 对特征进行标准化
    for i in range(1, 13):
        features_new.append(Normalization(features[:, i]))
    features_new.append(features[:, 13])
    features_new = np.array(features_new).T

    A, B, C = grade_10(features_new)
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    return A, B, C


def Normalization(sqi):
    """
    对特征进行[-1,1]标准化
    :param sqi:
    :return:
    """
    norm_sqi = 2 * ((sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi))) - 1

    return norm_sqi


def grade_30(features):
    """
    按不同类别进行划分
    :param features:
    :return:
    """
    A1 = []
    A2 = []
    B1 = []
    B2 = []
    C = []
    for i in range(0, len(features)):
        if features[i, 13] == 1:
            A1.append(features[i, :])
        elif features[i, 13] == 2:
            A2.append(features[i, :])
        elif features[i, 13] == 3:
            B1.append(features[i, :])
        elif features[i, 13] == 4:
            B2.append(features[i, :])
        else:
            C.append(features[i, :])

    return A1, A2, B1, B2, C


def test_30(filepath, filename):
    """
    去除特征值中的nan和inf等异常值
    :param filepath:
    :param filename:
    :return:
    """
    features = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
    print(np.shape(features))

    # 去除特征中的异常值（nan和inf）
    nan_con = np.where(np.isnan(features))[0]
    print("文件%s的inf位置为：" % filename, nan_con)
    features = np.delete(features, nan_con, axis=0)  # 执行删除 只删除了第2个数据
    # print(np.shape(features))

    inf_con = np.where(np.isinf(features))[0]
    print("文件%s的nan位置为：" % filename, inf_con)
    features = np.delete(features, inf_con, axis=0)  # 执行删除 只删除了第2个数据
    # print(np.shape(features))

    features_new = []
    features_new.append(features[:, 0])
    # 对特征进行标准化
    for i in range(1, 13):
        features_new.append(Normalization(features[:, i]))
    features_new.append(features[:, 13])
    features_new = np.array(features_new).T

    A1, A2, B1, B2, C = grade_30(features_new)
    A1 = np.array(A1)
    A2 = np.array(A2)
    B1 = np.array(B1)
    B2 = np.array(B2)
    C = np.array(C)

    return A1, A2, B1, B2, C


if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\Result_new\result_30/"
    filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    scale = 30  # 信号质量评估的时间尺度

    for filename in filenames:
        if scale == 10:  # 选择10s尺度
            A, B, C = test_10(filepath, filename)
            # 保存划分结果
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_10/A.txt", 'a+') as f:
                np.savetxt(f, A, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_10/B.txt", 'a+') as f:
                np.savetxt(f, B, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_10/C.txt", 'a+') as f:
                np.savetxt(f, C, fmt='%.5f', delimiter='    ')
        elif scale == 30:
            A1, A2, B1, B2, C = test_30(filepath, filename)
            # 保存划分结果
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_30/A1.txt", 'a+') as f:
                np.savetxt(f, A1, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_30/A2.txt", 'a+') as f:
                np.savetxt(f, A2, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_30/B1.txt", 'a+') as f:
                np.savetxt(f, B1, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_30/B2.txt", 'a+') as f:
                np.savetxt(f, B2, fmt='%.5f', delimiter='    ')
            with open(r"E:\研一\信号质量评估\Result_new\result_norm_30/C.txt", 'a+') as f:
                np.savetxt(f, C, fmt='%.5f', delimiter='    ')
