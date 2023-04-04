"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: Normalization_single.py
 @Function: 对单个个体进行标准化
 @DateTime: 2022/11/11 11:26
 @SoftWare: PyCharm
"""
import numpy as np
import os
from sklearn import preprocessing

def test(filepath, filename):
    """
    去除特征值中的nan和inf等异常值
    :param filepath:
    :param filename:
    :return:
    """
    features = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
    num = np.shape(features)[1] - 1
    print(np.shape(features))

    # # 去除特征中的异常值（nan和inf）
    # nan_con = np.where(np.isnan(features))[0]
    # print("文件%s的nan位置为：" % filename, nan_con)
    # features = np.delete(features, nan_con, axis=0)  # 执行删除
    # # print(np.shape(features))

    # 补充特征中的异常值（nan和inf）用同类别平均值填充
    nan_con = np.where(np.isnan(features))
    features_pass = np.delete(features, nan_con, axis=0)  # 执行删除
    kind_num = np.unique(features[nan_con[0], num])
    mean_same = []
    # for k in kind_num:
    #     print("k = ", k)
    #     same_index = np.argwhere(features_pass[:, 17] == k)
    #     mean_same.append(np.mean(features_pass[same_index, :], axis=0))
    # print(np.shape(mean_same))
    for i in range(0, len(nan_con[0])):
        # print(features[nan_con[0][i], nan_con[1][i]])
        for k in kind_num:
            same_index = np.argwhere(features_pass[:, num] == k)
            mean_same = (np.mean(features_pass[same_index, :], axis=0))
            if features[nan_con[0][i], num] == k:
                # print("k = ", k)
                features[nan_con[0][i], nan_con[1][i]] = mean_same[0][nan_con[1][i]]
                # print(mean_same[0][nan_con[1][i]])
            # elif features[nan_con[0][i], 17] == 2:
            #     features[nan_con[0][i], nan_con[1][i]] = mean_same[1][0][nan_con[1][i]]
            #     print(mean_same[1][0][nan_con[1][i]])
            # elif features[nan_con[0][i], 17] == 3:
            #     features[nan_con[0][i], nan_con[1][i]] = mean_same[2][0][nan_con[1][i]]
            #     print(mean_same[2][0][nan_con[1][i]])
    # nan_con = np.where(np.isnan(features))
    print("文件%s的nan位置为：" % filename, nan_con)
    features = np.delete(features, nan_con, axis=0)  # 执行删除
    # print(np.shape(features))

    inf_con = np.where(np.isinf(features))[0]
    print("文件%s的inf位置为：" % filename, inf_con)
    features = np.delete(features, inf_con, axis=0)  # 执行删除
    # print(np.shape(features))

    features_new = []
    features_new.append(features[:, 0])
    # 对特征进行标准化
    for i in range(1, num):
        features_new.append(Normalization(features[:, i]))
    features_new.append(features[:, num])
    features_new = np.array(features_new).T

    return features_new


def Normalization(sqi):
    """
    对特征进行[-1,1]标准化
    :param sqi:
    :return:
    """

    norm_sqi = preprocessing.scale(sqi)     # Z-score标准化
    # norm_sqi = (sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi))      # 归一化

    return norm_sqi


if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\result4.1\10s/"
    filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    # filenames = ['282.txt']  # 获取该文件夹中所有样本的特征结果
    # scale = 30  # 信号质量评估的时间尺度
    for filename in filenames:
            feature_new = test(filepath, filename)
            # with open(os.path.join(r"E:\研一\信号质量评估\Result_new\result_norm_10/", filename)) as f:
            np.savetxt(os.path.join(r"E:\研一\信号质量评估\result4.1\delete_nan_inf4.1\norm_10s/", filename), feature_new, fmt='%.6f', delimiter='    ')
