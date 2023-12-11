"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: classification_10.py
 @Function: 分级分类
 @DateTime: 2023/4/22 10:26
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


# 混淆矩阵
def cmpic(y_true, y_pred, i):
    def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['A1', 'A2', 'B1', 'B2', 'C']):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(len(cm))
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.05, hspace=0.5)
    if (i == 1):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: RF')
    if (i == 2):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: DT')
    if (i == 3):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: KNN')
    if (i == 4):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: XGboost')
    if (i == 5):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: lgb')
    if (i == 6):
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: Cb')


if __name__ == '__main__':
    # 数据文件路径
    filepath = r"E:\研一\信号质量评估\result5.21\normalization\30s/"
    saveroot = r"E:\研一\信号质量评估\result5.21\normalization\未调参/"
    label = ('RF', 'DT', 'KNN', 'XG', 'Lgb', 'Cb')
    # filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')
    # filenames = [282]
    # filenames = filenames[35:]
    for filename in filenames:
        pre_score = []
        file_num = int(filename)
        filename = str(int(filename))
        feature_file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
        X = feature_file[:, 1:20]
        y = feature_file[:, 20]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        # # 将体动占比转换为0,1,2三种可能
        X[(X[:, 14] >= 0.5), 14] = 2
        X[(0 < X[:, 14]) * (X[:, 14] <= 0.5), 14] = 1

        print("%s的结果为：" % filename)
        # strKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 分层交叉验证
        # # strKFold = ShuffleSplit(train_size=.5,test_size=.4,n_splits=10)   # 打乱后随机分配
        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 使用十折交叉验证
        kfold = strtfdKFold.split(X, y)

        for k, (train, test) in enumerate(kfold):
            y_pre_final, y_test_final = [], []
            print("第 %d 折" % (k + 1))
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            #
            RF = RandomForestClassifier(n_estimators=500, random_state=42)
            RF.fit(X_train, y_train)
            score = RF.score(X_test, y_test)
            print("分类的准确率为：", score)
            pre_score.append(score)
            continue

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
            # y_test_final, y_pre_final = [], []

            # print(np.where(X_train[:, 14] == 1)[0])
            # ---------------------------分出C1（不使用分类器）-----------------------------
            print(len(y_test))
            c1_index = []
            c1_train_index = np.where(X_train[:, 14] == 2)[0]
            for num_c1 in range(0, len(y_test)):
                if X_test[num_c1, 14] == 2:  # 直接将体动占比超过50%的片段判为C级（C1）
                    y_pre_final.append(5)
                    y_test_final.append(int(y_test[num_c1]))
                    c1_index.append(num_c1)
            print(y_pre_final)
            print(y_test_final)
            # print(len(c1_index))
            X_train = np.delete(X_train, c1_train_index, axis=0)  # 执行删除训练集中体动占比大于50%的标签
            y_train = np.delete(y_train, c1_train_index, axis=0)  # 执行删除训练集中体动占比大于50%的标签
            X_test = np.delete(X_test, c1_index, axis=0)  # 执行删除c1
            y_test = np.delete(y_test, c1_index, axis=0)  # 执行删除c1
            # print(len(y_test))

            # -----------C1已全部去除，开始利用分类器进行级联分类分出C2（使用ASQI1、KSQI、SNR、Sampen）-----------
            c2_train_index = np.where(y_train == 5)[0]
            # 仅保留训练集中所有C级特征的均值和中位数作为最后的两个C标签，下一步进行A和B二分类
            average_c2 = np.mean(X_train[c2_train_index, :], axis=0)  # 用所有c2的平均值作为第一条C标签
            median_c2 = np.median(X_train[c2_train_index, :], axis=0)  # 用所有c2的中位数作为第二条C标签
            if len(y_test):
                c2_indx = []
                RF = RandomForestClassifier(n_estimators=500, random_state=42)
                RF.fit(X_train[:, (5, 11, 13, 15)], y_train)
                y_pred_c2 = np.round(RF.predict(X_test[:, (5, 11, 13, 15)]))
                for num_c2 in range(0, len(y_test)):
                    if y_pred_c2[num_c2] == 5:
                        y_pre_final.append(5)
                        y_test_final.append(int(y_test[num_c2]))
                        c2_indx.append(num_c2)
                print(y_pre_final)
                print(y_test_final)
                X_train = np.delete(X_train, c2_train_index, axis=0)  # 执行删除训练集中所有c级
                y_train = np.delete(y_train, c2_train_index, axis=0)  # 执行删除训练集中所有c级
                X_test = np.delete(X_test, c2_indx, axis=0)  # 执行删除c2
                y_test = np.delete(y_test, c2_indx, axis=0)  # 执行删除c2
            # # 仅保留训练集中剩下所有C2的均值和中位数作为仅剩的2条C级标签
            # X_train = np.vstack((X_train, average_c2))
            # X_train = np.vstack((X_train, median_c2))
            # y_train = np.append(y_train, [5, 5])

            # ------------------------先利用其余特征分出A1+A2，B1+B2（先二分类A B，再二分类1 2）------------------------------
            # 将剩余的标签转换为A和B两类
            new_ytrain = []
            for i in range(0, len(y_train)):
                if y_train[i] == 1 or y_train[i] == 2:
                    new_ytrain.append(0)
                elif y_train[i] == 3 or y_train[i] == 4:
                    new_ytrain.append(1)
                else:
                    new_ytrain.append(2)

            if len(y_test):
                B1_index = []
                RF = RandomForestClassifier(n_estimators=500, random_state=42)
                RF.fit(X_train[:, (0, 1, 2, 3, 6, 4, 7, 8, 9, 10, 12, 16, 17, 18)], new_ytrain)
                y_pred_b1 = np.round(RF.predict(X_test[:, (0, 1, 2, 3, 6, 4, 7, 8, 9, 10, 12, 16, 17, 18)]))
                for num_b1 in range(0, len(y_test)):
                    if y_pred_b1[num_b1] == 0 and X_test[num_b1, 14] == 0:
                        y_pre_final.append(1)
                        y_test_final.append(int(y_test[num_b1]))
                    elif y_pred_b1[num_b1] == 0 and X_test[num_b1, 14] == 1:
                        y_pre_final.append(2)
                        y_test_final.append(int(y_test[num_b1]))
                    elif y_pred_b1[num_b1] == 1 and X_test[num_b1, 14] == 0:
                        y_pre_final.append(3)
                        y_test_final.append(int(y_test[num_b1]))
                    elif y_pred_b1[num_b1] == 1 and X_test[num_b1, 14] == 1:
                        y_pre_final.append(4)
                        y_test_final.append(int(y_test[num_b1]))
                    else:
                        y_pre_final.append(5)
                        y_test_final.append(int(y_test[num_b1]))
                        # B1_index.append(num_b1)
                print(y_pre_final)
                print(y_test_final)
            #     X_test = np.delete(X_test, B1_index, axis=0)  # 执行删除b1
            #     y_test = np.delete(y_test, B1_index, axis=0)  # 执行删除b1
            #
            # # ---------------------------分出B2(使用剩余的标签)---------------------------
            # if len(y_test):
            #     B2_index = []
            #     RF = RandomForestClassifier(n_estimators=500, random_state=42)
            #     RF.fit(X_train[:, (2, 3, 4, 7, 8, 9, 10, 12, 15, 16, 17)], y_train)
            #     y_pred_b2 = np.round(RF.predict(X_test[:, (2, 3, 4, 7, 8, 9, 10, 12, 15, 16, 17)]))
            #     for num_b2 in range(0, len(y_test)):
            #         if y_pred_b2[num_b2] == 2:
            #             y_pre_final.append(2)
            #             y_test_final.append(int(y_test[num_b2]))
            #             B2_index.append(num_b2)
            #     print(len(y_pre_final))
            #     print(len(y_test_final))
            #     X_test = np.delete(X_test, B2_index, axis=0)  # 执行删除b2
            #     y_test = np.delete(y_test, B2_index, axis=0)  # 执行删除b2
            #     print(len(y_test))
            #
            # # ---------------------------剩下的全是A---------------------------
            # if len(y_test):
            #     for num_a in range(0, len(y_test)):
            #         y_pre_final.append(1)
            #         y_test_final.append(int(y_test[num_a]))
            # print(y_pre_final)
            # print(y_test_final)

            # ---------------------------计算最终的分类性能---------------------------
            print(np.sum((np.array(y_pre_final) - np.array(y_test_final)) == 0) / len(y_test_final))
            pre_score.append(np.sum((np.array(y_pre_final) - np.array(y_test_final)) == 0) / len(y_test_final))

            # plt.figure()
            # a = cmpic(y_test_final, y_pre_final, 1)
            # plt.show()

        #     class_names = ['A', 'B', 'C']
        #     print("1、随机森林的性能评估：\n", classification_report(y_test_final, y_pre_final, target_names=class_names))
        # print("十折交叉验证的平均准确度为：", np.mean(pre_score))
        score_result = [(file_num, np.mean(pre_score))]
        with open(os.path.join(saveroot, 'RF_未级联.txt'), 'a+') as file:
            np.savetxt(file, score_result, fmt='%.4f\t', delimiter=' ')

        # result = list(zip(y_test, y_pre_c1, y_pred_c2, y_pred_b1, y_pred_b
        #         # with open(r"E:\研一\信号质量评估\result4.21\test.txt", 'a+', encoding='utf-8') as file:
        #         #     np.savetxt(file, result, fmt='%.0f\t', delimiter='\t')2))
        # print(y_pre)
