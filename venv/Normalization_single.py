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

    # 补充特征中的nan值用同类别的平均值填充
    nan_con = np.where(np.isnan(features))
    features_pass1 = np.delete(features, nan_con, axis=0)  # 执行删除，以此作为过渡数据
    kind_num1 = np.unique(features[nan_con[0], num])
    mean_same = []
    # for k in kind_num:
    #     print("k = ", k)
    #     same_index = np.argwhere(features_pass[:, 17] == k)
    #     mean_same.append(np.mean(features_pass[same_index, :], axis=0))
    # print(np.shape(mean_same))
    for i in range(0, len(nan_con[0])):
        # print(features[nan_con[0][i], nan_con[1][i]])
        for k in kind_num1:
            same_index1 = np.argwhere(features_pass1[:, num] == k)
            mean_same1 = (np.mean(features_pass1[same_index1, :], axis=0))
            if features[nan_con[0][i], num] == k:
                # print("k = ", k)
                features[nan_con[0][i], nan_con[1][i]] = mean_same1[0][nan_con[1][i]]
                # print(mean_same[0][nan_con[1][i]])
            # elif features[nan_con[0][i], 17] == 2:
            #     features[nan_con[0][i], nan_con[1][i]] = mean_same[1][0][nan_con[1][i]]
            #     print(mean_same[1][0][nan_con[1][i]])
            # elif features[nan_con[0][i], 17] == 3:
            #     features[nan_con[0][i], nan_con[1][i]] = mean_same[2][0][nan_con[1][i]]
            #     print(mean_same[2][0][nan_con[1][i]])
    # nan_con = np.where(np.isnan(features))
    print("文件%s的nan位置为：" % filename, nan_con)
    # features = np.delete(features, nan_con, axis=0)  # 执行删除
    # print(np.shape(features))

    inf_con = np.where(np.isinf(features))
    features_pass2 = np.delete(features, inf_con, axis=0)  # 执行删除，以此作为过渡数据
    kind_num2 = np.unique(features[inf_con[0], num])
    for i2 in range(0, len(inf_con[0])):
        # print(features[nan_con[0][i], nan_con[1][i]])
        for k2 in kind_num2:
            same_index2 = np.argwhere(features_pass2[:, num] == k2)
            mean_same2 = (np.mean(features_pass2[same_index2, :], axis=0))
            if features[inf_con[0][i2], num] == k2:
                features[inf_con[0][i2], inf_con[1][i2]] = mean_same2[0][inf_con[1][i2]]
    print("文件%s的inf位置为：" % filename, inf_con)
    # features = np.delete(features, inf_con, axis=0)  # 执行删除
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

    # norm_sqi = preprocessing.scale(sqi)     # Z-score标准化
    norm_sqi = (sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi))      # 归一化
    # norm_sqi = sqi      # 归一化

    return norm_sqi


if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\result11.8\10s/"
    filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')
    # filenames = ['229', '286', '1354']  # 获取该文件夹中所有样本的特征结果
    # scale = 30  # 信号质量评估的时间尺度
    for filename in filenames:
        # filename = filename + '.txt'
        feature_new = test(filepath, filename)
        # with open(os.path.join(r"E:\研一\信号质量评估\Result_new\result_norm_10/", filename)) as f:
        np.savetxt(os.path.join(r"E:\研一\信号质量评估\result11.8\normalization/10s/", filename), feature_new, fmt='%.6f\t', delimiter='\t')
