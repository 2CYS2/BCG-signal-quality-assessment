"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: classification_10s(共性特性特征级联分类).py
 @Function: 利用筛选出的共性特性特征进行级联分类【结构类似stacking集成学习方法，略有不同】
 @DateTime: 2023/10/16 19:37
 @SoftWare: PyCharm
"""

from matplotlib import pyplot
from sklearn import tree
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, ShuffleSplit, \
    GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier
from sklearn import metrics
import joblib

# 混淆矩阵
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
    # plt.savefig(r"E:\研一\信号质量评估\result5.8\normalization\留一法\10s\plot/" + filename + '.png')


if __name__ == '__main__':
    # 数据文件路径
    filepath = r"E:\研一\信号质量评估\result12.5\normalization\10s/"
    saveroot = r"E:\研一\信号质量评估\result12.5\normalization\留一法(DL)--特征级联(特征贡献度排序5)/"
    saveroot_particular = saveroot + "特性特征分类结果/"
    # for k in range(0, 40):
    #     print("------------------第 %.0f 折-----------------" % (k + 1))
    #     filenames = os.listdir(filepath)
    #     # filenames_test = ['1006.txt']
    #     filenames_test = filenames[k:k + 1]
    #     file_num = int(filenames_test[0].split('.txt')[0])
    #     print(filenames_test)
    #     for i in filenames_test:
    #         filenames.remove(i)
    #     filenames_train = filenames
    #
    #     feature_train = np.loadtxt(os.path.join(filepath, filenames_train[0]), encoding='utf-8')
    #     for f1 in range(1, 39):
    #         # print(filenames_train[f1])
    #         feature_file = np.loadtxt(os.path.join(filepath, filenames_train[f1]), encoding='utf-8')
    #         feature_train = np.vstack((feature_train, feature_file))
    #     print("训练集为：", np.shape(feature_train))
    #     # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\train.txt", feature_train, fmt='%.4f\t')
    #
    #     feature_test = np.loadtxt(os.path.join(filepath, filenames_test[0]), encoding='utf-8')
    #     # feature_test = feature_test[feature_test[:, 19].argsort()]  # 将所有片段按质量标签降序排列
    #
    #     # 规则1：1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20
    #     # 规则2：1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20
    #     # 规则3：4, 8, 9, 11, 14, 16, 19, 20
    #     # 规则4：1，4, 8, 9, 11, 14, 16, 19
    #     # 规则5：1，4, 8, 9, 11, 14, 16, 19, 20
    #     # 按特征定义划分：
    #
    #     tag_num1 = feature_test.shape[1]-1
    #     X_train = feature_train[:, (1, 4, 8, 9, 11, 14, 16, 19, 20)]
    #     y_train = feature_train[:, tag_num1]
    #     X_test = feature_test[:, (1, 4, 8, 9, 11, 14, 16, 19, 20)]
    #     y_test = feature_test[:, tag_num1]
    #     # ---------------------------开始分类-----------------------------
    #     # ---------------------------排除受体动影响的C1(没有使用分类器)-----------------------------
    #     c1_train_index = np.where(feature_train[:, 15] == 1)[0]   # 训练集中包含体动的索引
    #     X_train = np.delete(X_train, c1_train_index, axis=0)  # 执行删除训练集中的体动
    #     y_train = np.delete(y_train, c1_train_index, axis=0)  # 执行删除训练集中的体动
    #
    #     c1_test_index = np.where(feature_test[:, 15] == 1)[0]   # 测试集中包含体动的索引
    #     print("受体动影响的C级数量为：", len(c1_test_index))
    #     X_test = np.delete(X_test, c1_test_index, axis=0)  # 执行删除测试集中c1
    #     y_test = np.delete(y_test, c1_test_index, axis=0)  # 执行删除测试集中c1
    #
    #     # ---------------------------利用特性特征得到初始结果，并保存相应结果至最后一个特征---------------------------
    #     # ---------------------第一个学习器：RF-----------------------
    #     print("第一个学习器：RF")
    #     RF = RandomForestClassifier(n_estimators=500, random_state=42)
    #     RF.fit(X_train, y_train)
    #     y_pred_particular = RF.predict(X_test)
    #     print("测试集的特征形状为：", np.shape(X_test))
    #     print("特性特征的结果形状为：", np.shape(y_pred_particular))
    #     print("测试集标签的形状为：", np.shape(y_test))
    #     for c1_index in c1_test_index:
    #         y_pred_particular = np.insert(y_pred_particular, c1_index, 3, axis=None)
    #     print("特性特征的结果形状为：", np.shape(y_pred_particular))
    #
    #     # # ---------------------第二个学习器：XGboost-----------------------
    #     print("第二个学习器：XGboost")
    #     model = XGBClassifier(n_estimators=400, random_state=42)
    #     model.fit(X_train, y_train)
    #     y_pred_particular1 = model.predict(X_test)
    #     print("测试集的特征形状为：", np.shape(X_test))
    #     print("特性特征的结果形状为：", np.shape(y_pred_particular1))
    #     print("测试集标签的形状为：", np.shape(y_test))
    #     for c1_index in c1_test_index:
    #         y_pred_particular1 = np.insert(y_pred_particular1, c1_index, 3, axis=None)
    #     print("特性特征的结果形状为：", np.shape(y_pred_particular1))
    #
    #     # 规则1：0, 6, 7, 10, 12, 14, 15, 19, 20
    #     # 规则2：0, 6, 7, 10, 12, 14, 15, 20
    #     # 规则3：0, 6, 7, 10, 12, 15
    #     # 规则4：0, 6, 7, 10, 12, 15, 20
    #     # 规则5：0, 6, 7, 10, 12, 15, 20
    #     # 按特征定义划分：
    #     result1 = np.hstack((feature_test[:, (0, 6, 7, 10, 12, 15, 20)], y_pred_particular.reshape(-1, 1)))     # 添加第一个学习器的结果
    #     result1 = np.hstack((result1, y_pred_particular1.reshape(-1, 1)))      # 添加第二个学习器的结果
    #     result1 = np.hstack((result1, feature_test[:, tag_num1].reshape(-1, 1)))
    #     result1 = result1[result1[:, 0].argsort()]  # 将所有片段按质量标签序号降序排列
    #
    #     print("第一步级联中保存结果的形状为：", np.shape(result1))
    #     np.savetxt(os.path.join(saveroot_particular, filenames_test[0]), result1, '%.0f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.0f\t%.0f\t%.0f\t', delimiter='\t')

        # ---------------------------利用共性特征得到最终结果---------------------------
    for k in range(0, 40):
        print("------------------第 %.0f 折-----------------" % (k + 1))
        filenames = os.listdir(filepath)
        # filenames_test = ['1006.txt']
        filenames_test = filenames[k:k + 1]
        file_num = int(filenames_test[0].split('.txt')[0])
        print(filenames_test)
        for i in filenames_test:
            filenames.remove(i)
        filenames_train = filenames

        feature_train = np.loadtxt(os.path.join(saveroot_particular, filenames_train[0]), encoding='utf-8')
        for f1 in range(1, 39):
            # print(filenames_train[f1])
            feature_file = np.loadtxt(os.path.join(saveroot_particular, filenames_train[f1]), encoding='utf-8')
            feature_train = np.vstack((feature_train, feature_file))
        print(np.shape(feature_train))

        feature_test = np.loadtxt(os.path.join(saveroot_particular, filenames_test[0]), encoding='utf-8')
        # feature_test = feature_test[feature_test[:, 19].argsort()]  # 将所有片段按质量标签降序排列

    # 按特征贡献度划分：1, 2, 3, 4, 5, 7, 8, 9
    # 按特征定义划分：
    # 剔除部分特征后：1, 2, 3, 5, 6, 7, 8, 9
        tag_num = feature_test.shape[1]-1
        X_train = feature_train[:, (1, 2, 3, 4, 6, 7, 8)]
        y_train = feature_train[:, tag_num]
        X_test = feature_test[:, (1, 2, 3, 4, 6, 7, 8)]
        y_test = feature_test[:, tag_num]

        pre_score = []
        # ---------------------------排除C1(没有使用分类器)-----------------------------
        c1_train_index = np.where(feature_train[:, 5] == 1)[0]    # 训练集中包含体动的索引
        X_train = np.delete(X_train, c1_train_index, axis=0)  # 执行删除训练集中的体动
        y_train = np.delete(y_train, c1_train_index, axis=0)  # 执行删除训练集中的体动

        c1_test_index = np.where(feature_test[:, 5] == 1)[0]   # 测试集中包含体动的索引
        print(len(c1_test_index))
        X_test = np.delete(X_test, c1_test_index, axis=0)  # 执行删除测试集中c1
        y_test = np.delete(y_test, c1_test_index, axis=0)  # 执行删除测试集中c1

        # ---------------------------开始分类-----------------------------
        RF = RandomForestClassifier(n_estimators=500, random_state=42)
        RF.fit(X_train, y_train)
        y_pre_final = RF.predict(X_test)

        # model = XGBClassifier(n_estimators=400, random_state=42)
        # model.fit(X_train, y_train)
        # y_pre_final = model.predict(X_test)

        error_index = np.where(y_pre_final != y_test)[0] + 1
        for c1_index in c1_test_index:
            y_pre_final = np.insert(y_pre_final, c1_index, 3, axis=None)

        print(y_pre_final)
        print(feature_test[:, tag_num])
        print("错判的索引为：", error_index)
        print("错判的数量为：", len(error_index))
        print(np.sum((np.array(y_pre_final) - np.array(feature_test[:, tag_num])) == 0) / len(feature_test[:, tag_num]))
        pre_score.append(np.sum((np.array(y_pre_final) - np.array(feature_test[:, tag_num])) == 0) / len(feature_test[:, tag_num]))
        # #
        # # # 保存训练好的模型
        # # joblib.dump(RF1, 'D:\saved_model1/' + str(file_num) + '_1.pkl')
        # # joblib.dump(RF2, 'D:\saved_model1/' + str(file_num) + '_2.pkl')
        #
        # 保存分类混淆矩阵
        plot_matrix(feature_test[:, tag_num], y_pre_final, [1, 2, 3], title='Confusion_matrix', axis_labels=['a', 'b', 'c'])
        plt.savefig(saveroot + "plot/" + str(file_num) + '.png')
        #
        # # # # 输出各类别分类性能
        # # # class_names = ['A', 'B', 'C']
        # # # print("1、随机森林的性能评估：\n", classification_report(y_test_final, y_pre_final, target_names=class_names))
        # # # print("十折交叉验证的平均准确度为：", np.mean(pre_score))
        #
        # 保存分类准确率
        score_result = [(file_num, np.mean(pre_score))]
        with open(os.path.join(saveroot, '(XG+RF)+RF.txt'), 'a+') as file:
            np.savetxt(file, score_result, fmt='%.4f\t', delimiter=' ')

        # 保存分类错误的索引
        with open(os.path.join(saveroot + '\error_index/', str(file_num) + '.txt'), 'a+') as error_file:
            np.savetxt(error_file, error_index, fmt='%.0f\t', delimiter=' ')

        # 保存测试值与预测值
        result = list(zip(feature_test[:, tag_num], y_pre_final))
        with open(os.path.join(saveroot + '\\test&pre/', str(file_num) + '.txt'), 'a+') as result_file:
            np.savetxt(result_file, result, fmt='%.0f\t%.0f\t', delimiter='\t')

        # result = list(zip(y_test, y_pre_c1, y_pred_c2, y_pred_b1, y_pred_b
        #         # with open(r"E:\研一\信号质量评估\result4.21\test.txt", 'a+', encoding='utf-8') as file:
        #         #     np.savetxt(file, result, fmt='%.0f\t', delimiter='\t')2))
        # print(y_pre)
