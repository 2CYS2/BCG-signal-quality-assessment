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
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
    accuracy_score, classification_report
from sklearn import metrics
import random


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred)  # 生成混淆矩阵
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, fontsize=15)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels, fontsize=15)  # 将标签印在y轴坐标上
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] > 0:
                plt.text(j, i, format(cm[i][j], 'd'),
                         ha="center", va="center",
                         color="red" if cm[i][j] > thresh else "black", fontsize=20)  # 如果要更改颜色风格，需要同时更改此行
    # plt.show()


# # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
#     for i in range(np.shape(cm)[0]):
#         for j in range(np.shape(cm)[1]):
#             if int(cm[i][j] * 100 + 0.5) > 0:
#                 pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
#                         ha="center", va="center",
#                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# # 显示
#     pl.show()

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

    # # 1、将新标签的结果添加到特征集中
    # filepath = r"E:\研一\信号质量评估\result4.21\10s/"
    # sampenpath = r"E:\研一\信号质量评估\result4.21\quality_index/"
    # saveroot = r"E:\研一\信号质量评估\result4.21\new/"
    # # filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')
    # # filenames = [220, 221, 282, 286, 541, 549, 551, 560, 582, 586, 703, 704, 726, 952, 955, 961, 966, 967, 969, 1000, 1010]  # 获取该文件夹中所有样本的特征结果
    # # filenames = [551, 582, 735, 704, 221, 671, 955, 1354, 286, 560, 586, 952, 1000, 1010]
    # filenames = [967]
    # for filename in filenames:
    #     filename = str(int(filename))
    #     file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
    #     quality_data = np.loadtxt(os.path.join(r'E:\研一\信号质量评估\result4.21\quality_index/', filename + '.txt'), encoding='utf-8')
    #     # sampen = np.loadtxt(os.path.join(sampenpath, filename + '.txt'), encoding='utf-8')
    #     # print(sampen)
    #     # new_file = np.insert(file[:,0:15], 15, sampen[:, 1], axis=1)
    #     # new_file = np.insert(new_file, 16, sampen[:, 2], axis=1)
    #     # new_file = np.insert(new_file, 17, file[:, 15], axis=1)
    #
    #     # isqi2_e = ex(file[:, 4])
    #     # rsdsqi_e = ex(file[:, 5])
    #     # print(np.shape(isqi2_e))
    #     # new_file = np.insert(file[:, 0:19], 19, isqi2_e, axis=1)
    #     # new_file = np.insert(new_file, 18, rsdsqi_e, axis=1)
    #     # new_file = np.insert(new_file, 20, file[:, 19], axis=1)
    #
    #     # new_file = np.insert(file[:, 0:21], 21, sampen, axis=1)
    #     # new_file = np.insert(new_file, 21, file[:, 20], axis=1)
    #     new_file = np.insert(file[:, 0:19], 19, quality_data, axis=1)
    #
    #     print(np.shape(new_file))
    #     np.savetxt(os.path.join(saveroot, str(int(filename)) + '.txt'), new_file, fmt='%.6f\t')

    # 2、将质量标签转换为数字
    # scale = 10
    # filenames = [220, 221, 282, 286, 541, 549, 551, 560, 582, 586, 703, 704, 726, 952, 955, 961, 966, 967, 969, 1000, 1010]
    # for filename in filenames:
    #     quality_index = []
    #     filename = str(int(filename))
    #     quality_data = open(os.path.join(r'E:\BCG_data/', filename + '/' + '质量\SQ_label1_10s.txt'), 'r+', encoding='utf-8')
    #     quality_list = quality_data.readlines()  # 读取所有行的元素并返回一个列表
    #     for i in range(0, len(quality_list)):
    #         grade_num = GRADE(quality_list[i].split('\n')[0], scale)  # 将对应等级转换为数字，便于处理
    #         quality_index.append(grade_num)
    #         print(grade_num)
    #
    #     np.savetxt(os.path.join(r'E:\研一\信号质量评估\result4.21\quality_index/', filename + '.txt'), quality_index, fmt='%.0f')

    # 3、显著性检验
    # # 数据文件路径
    # filepath = r"E:\研一\信号质量评估\result4.1\delete_nan_inf4.1\norm_30s/"
    # filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')
    # feature_names = ['BSQI1', 'BSQI2', 'ISQI1', 'ISQI2', 'RsdSQI', 'ASQI', 'PSQI', 'CSQI', 'Avg_corr', 'SNR', 'SSQI',
    #                  'KSQI', 'artiafct', 'Sampen', 'ASQI2', 'Rsd_corr', 'ISQI2_exp', 'RsdSQI_exp', 'ASQI2new']
    # # filenames = [220]
    # for filename in filenames:
    #     print("正在处理%s" % filename + '.txt')
    #     file_num = int(filename)
    #     filename = str(int(filename))
    #     feature_file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
    #     for i in range(1, 20):
    #         statistic, pvalue = stats.mannwhitneyu(feature_file[:, i], feature_file[:, 20])
    #         print("%s的显著性检验结果为：statistic = %f, pvalue = %f" % (feature_names[i-1], statistic, pvalue))

    # # 4、统计各质量等级的总数
    # filepath = r"E:\BCG_data/"
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')
    # num, A, B, C = [], [], [], []
    # for filename in filenames:
    #     print("正在处理%d" % filename)
    #     num.append(filename)
    #     filename = np.str(int(filename))
    #     quality_data = open(os.path.join(r'E:\BCG_data/', filename + '/' + '质量\SQ_label1_10s.txt'), 'r+', encoding='utf-8')
    #     quality_list = quality_data.readlines()  # 读取所有行的元素并返回一个列表
    #     a, b, c = 0, 0, 0
    #     for i in quality_list:
    #         quality_index = i.split('\n')[0]
    #         if quality_index == 'a':
    #             a += 1
    #         elif quality_index == 'b':
    #             b += 1
    #         elif quality_index == 'c':
    #             c += 1
    #     print(a)
    #     print(b)
    #     print(c)
    #     A.append(a)
    #     B.append(b)
    #     C.append(c)
    #
    # result = list(zip(num, A, B, C))
    # with open(r"E:\研一\信号质量相关性统计\小论文\结果汇总5.17\5.18更新\quality_index_num.txt", 'a+') as file:
    #     np.savetxt(file, result, fmt='%.0f\t')

    # # 5、将对应数字转换为质量等级
    # quality_num = np.loadtxt(r"E:\研一\信号质量评估\result5.17\10s\671.txt", encoding='utf-8')[:, 19]
    # quality_data = []
    # print(quality_num)
    # for i in range(0, len(quality_num)):
    #     if quality_num[i] == 1:
    #         quality_data.append('a')
    #     elif quality_num[i] == 2:
    #         quality_data.append('b')
    #     elif quality_num[i] == 3:
    #         quality_data.append('c')
    #
    # print(quality_data)
    # np.savetxt(r"E:\BCG_data\671\质量\quality_10s.txt", quality_data, fmt='%s')

    # # 6、对比纠正前后各质量等级的总数
    # filepath = r"E:\BCG_data/"
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')
    # for filename in filenames:
    #     # print("正在处理%d" % filename)
    #     filename = np.str(int(filename))
    #     quality_data_new = open(os.path.join(r'E:\BCG_data/', filename + '/' + '质量\SQ_label1_10s.txt'), 'r+', encoding='utf-8')
    #     quality_data_old = open(os.path.join(r'E:\BCG_data/', filename + '/' + '质量\old\SQ_label1_10s.txt'), 'r+', encoding='utf-8')
    #
    #     quality_list_new = quality_data_new.readlines()  # 读取所有行的元素并返回一个列表
    #     quality_list_old = quality_data_old.readlines()  # 读取所有行的元素并返回一个列表
    #     if len(quality_list_new) != len(quality_list_old):
    #         print("纠正前后不相等！！！！！")
    #     else:
    #         print("纠正前后相等。")

    # # 7、计算分类性能（pre，recall，f1-score）
    # filepath = r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(特征贡献度排序)\test&pre/"
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num_new.txt", encoding='utf-8')
    # # filenames = os.listdir(filepath)
    # precision, recall, f1, file_num = [], [], [], []
    # # filenames = ['220.txt', '584.txt']
    # king_key = ['1.0', '2.0', '3.0']
    # result_key = ['precision', 'recall', 'f1-score']
    # result = np.zeros((40, 9))
    # pre = np.zeros((2, 2))
    # for num in range(40):
    #     filename = str(int(filenames[num])) + '.txt'
    #     print("正在处理%s" % filename)
    #     data = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
    #     actual, predicted = data[:, 0], data[:, 1]
    #     p3 = precision_score(actual, predicted, average='weighted')
    #     r3 = recall_score(actual, predicted, average='weighted')
    #     f1score3 = f1_score(actual, predicted, average='weighted')
    #     # cp_score = matthews_corrcoef(actual, predicted)     # 用于替代简单的ACC
    #     # MCC = cohen_kappa_score(actual, predicted)      # 不受类别不均衡的影响
    #     print("精确率为：%.2f" % (p3 * 100))
    #     print("召回率为：%.2f" % (r3 * 100))
    #     print("F1-score为：%.2f" % (f1score3 * 100))
    #     # print("cp-score：%.2f" % (cp_score*100))
    #     # print("MCC为：%.2f" % (MCC*100))
    #     file_num.append(int(filename.split('.txt')[0]))
    #     precision.append(p3)
    #     recall.append(r3)
    #     f1.append(f1score3)
    #     report = classification_report(actual, predicted, output_dict=True)
    #     print(report)
    #     for i in range(3):
    #         for j in range(3):
    #             result[num][i*3+j] = report[king_key[i]][result_key[j]]
    #     print(result)
    #     pre = np.vstack((pre, data))
    # # 根据真实标签与预测标签绘制混淆矩阵
    # plot_matrix(pre[2:,0], pre[2:,1], [1, 2, 3], title='Confusion_matrix', axis_labels=['a', 'b', 'c'])
    # plt.savefig(r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(特征贡献度排序)\Confusion_matrix.png")
    # plt.show()
    # preform = list(zip(file_num, precision, recall, f1))
    # with open(r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL方法)\perform总.txt", 'a+') as file:
    #     np.savetxt(file, preform, fmt='%.4f', delimiter='\t')
    # with open(r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(按特征定义排序)\perform.txt", 'a+') as file:
    #     np.savetxt(file, result, fmt='%.4f', delimiter='\t')

    # # # 8、绘制BCG信号及其J峰
    # filepath = r"E:\non_Align/"
    # filenames = [1303]
    # for filename in filenames:
    #     bcg = pd.read_csv(os.path.join(filepath, str(int(filename))+'/'+'DSbcg_sig_1000hz3.txt'), encoding='utf-8').to_numpy()
    #     tm = np.loadtxt(os.path.join(filepath, str(int(filename))+'predict_J.txt'), encoding='utf-8')
    #     rpeaks = np.loadtxt(os.path.join(filepath, str(int(filename))+'/'+'final_Rpeak.txt'), encoding='utf-8')
    #     rr = np.diff(rpeaks)
    #     up = np.where(rr > 2000)[0]
    #     rr = np.delete(rr, up, axis=0)
    #     down = np.where(rr < 500)[0]
    #     rr = np.delete(rr, down, axis=0)
    #     print(60 * 1000/np.mean(rr))
    #     # tm = tm.astype('int64')
    #     # plt.figure()
    #     # plt.plot(bcg)
    #     # plt.plot(tm, bcg[tm], 'o')
    #     # plt.show()

    # # 9、查找被错判为C的A级片段
    # data = np.loadtxt(r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(特征贡献度排序)\test&pre\1308.txt", encoding='utf-8')
    # for i in range(len(data[:, 0])):
    #     if data[i, 0] == 2 and data[i, 1] == 3:
    #         print(i+1, end=', ')

    # 10、个体内对特征重要性进行排序
    result = np.loadtxt(r"E:\研一\信号质量评估\result11.8\normalization\个体内训练分类(获取特征重要性)/RF.txt", encoding='utf-8')
    num = np.arange(1, result.shape[1] + 1)
    data = np.zeros((result.shape[0] * 2, result.shape[1]))
    zeros_data = data
    for i in range(result.shape[0]):
        data[2 * i, :] = num
        data[2 * i + 1, :] = np.argsort(result[i, :])[::-1] + 1
        # print(num)
        # print(np.argsort(result[i, :])[::-1] + 1)
    # print(data)
    # np.savetxt(r"E:\研一\信号质量评估\result11.8\normalization\个体内训练分类(获取特征重要性)\sort.txt", data, fmt='%.0f', delimiter='\t')

    for j in range(result.shape[0]):
        zeros_data[2 * j, :] = num[np.argsort(data[2 * j + 1, :])]      # 该特征在这一行的排序
        zeros_data[2 * j + 1, :] = data[2 * j + 1, :][np.argsort(data[2 * j + 1, :])]   # 第几个特征
    print(zeros_data)
    np.savetxt(r"E:\研一\信号质量评估\result11.8\normalization\个体内训练分类(获取特征重要性)\sort.txt", data, fmt='%.0f', delimiter='\t')