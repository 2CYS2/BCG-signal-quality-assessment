"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: classification_person_10s.py
 @Function: 30s尺度下，个体间做留一法
 @DateTime: 2023/4/24 15:44
 @SoftWare: PyCharm
"""
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
    filepath = r"E:\研一\信号质量评估\result5.21\normalization\30s/"
    saveroot = r"E:\研一\信号质量评估\result5.21\normalization\留一法\30s/"
    # filepath2 = np.loadtxt(r'E:\研一\信号质量评估\Result_new\result_norm_10/B.txt')
    # filepath3 = np.loadtxt(r'E:\研一\信号质量评估\Result_new\result_norm_10/C.txt')
    # filepath = np.vstack((filepath1,filepath2,filepath3))
    # filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')
    # filenames = [220, 221, 282, 286, 541, 549, 551, 560, 582, 586, 703, 704, 726, 952, 955, 961, 966, 967, 969, 1000, 1010]
    for k in range(0, 40):
        print("------------------第 %.0f 折-----------------" % (k + 1))
        filenames = os.listdir(filepath)
        # filenames_test = ['1004.txt']
        filenames_test = filenames[k:k + 1]
        # filenames_test = ['971.txt', '935.txt', '966.txt', '541.txt']
        # filenames_test = ['967.txt']
        file_num = int(filenames_test[0].split('.txt')[0])
        print(filenames_test)
        for i in filenames_test:
            filenames.remove(i)
        filenames_train = filenames

        feature_train = np.loadtxt(os.path.join(filepath, filenames_train[0]), encoding='utf-8')
        for f1 in range(1, 39):
            # print(filenames_train[f1])
            feature_file = np.loadtxt(os.path.join(filepath, filenames_train[f1]), encoding='utf-8')
            feature_train = np.vstack((feature_train, feature_file))
        print(np.shape(feature_train))
        # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\train.txt", feature_train, fmt='%.4f\t')

        feature_test = np.loadtxt(os.path.join(filepath, filenames_test[0]), encoding='utf-8')
        # for f2 in range(1, 4):
        #     # print(filenames_test[f2])
        #     feature_file = np.loadtxt(os.path.join(filepath, filenames_test[f2]), encoding='utf-8')
        #     feature_test = np.vstack((feature_test, feature_file))
        # print(np.shape(feature_test))
        # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\test.txt", feature_test, fmt='%.4f\t')

        # feature_verify = np.loadtxt(os.path.join(filepath, filenames_verify[0]), encoding='utf-8')
        # for f3 in range(1, 8):
        #     print(filenames_train[f3])
        #     feature_file = np.loadtxt(os.path.join(filepath, filenames_verify[f3]), encoding='utf-8')
        #     feature_verify = np.vstack((feature_verify, feature_file))
        # print(np.shape(feature_verify))
        # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\verify.txt", feature_verify, fmt='%.4f\t')

        # X = feature_file[:, 1:12]
        # y = feature_file[:,12]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        # X_train, X_verify, y_train, y_verify = train_test_split(X, y, test_size=0.25, random_state=2)
        feature_test = feature_test[feature_test[:, 20].argsort()]  # 将所有片段按质量标签降序排列,方便后续统计错判的片段

        # # 将体动占比转换为0,1,2三种可能值
        feature_train[(feature_train[:, 15] >= 0.5), 15] = 2
        feature_train[(0 < feature_train[:, 15]) * (feature_train[:, 15] <= 0.5), 15] = 1
        feature_test[(feature_test[:, 15] >= 0.5), 15] = 2
        feature_test[(0 < feature_test[:, 15]) * (feature_test[:, 15] <= 0.5), 15] = 1

        X_train = feature_train[:, 1:20]
        y_train = feature_train[:, 20]
        X_test = feature_test[:, 1:20]
        y_test = feature_test[:, 20]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        # print("%s的结果为：" % filenames)
        # # strKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 分层交叉验证
        # # # strKFold = ShuffleSplit(train_size=.5,test_size=.4,n_splits=10)   # 打乱后随机分配
        # strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # kfold = strtfdKFold.split(X, y)
        #
        # for k, (train, test) in enumerate(kfold):
        #     y_pre_final, y_test_final = [], []
        #     print("第 %d 折" % (k+1))
        #     X_train = X[train]
        #     y_train= y[train]
        #     X_test = X[test]
        #     y_test = y[test]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        y_test_final, y_pre_final, pre_score = [], [], []
        error_index = []  # 保存错判的索引值
        error = []  # 保存错判的真实索引值

        # print(np.where(X_train[:, 14] == 1)[0])
        # ---------------------------分出C1(没有使用分类器)-----------------------------
        print(len(y_test))
        c1_index = []
        c1_train_index = np.where(X_train[:, 14] == 2)[0]
        for num_c1 in range(0, len(y_test)):
            if X_test[num_c1, 14] == 2:  # 直接将受体动影响的片段判为C级（C1）
                y_pre_final.append(5)
                y_test_final.append(int(y_test[num_c1]))
                c1_index.append(num_c1)
                if int(y_test[num_c1]) != 5:  # 错判则记录序号
                    error_index.append(num_c1)
                    error.append(feature_test[num_c1, 0])
        print(y_pre_final)
        print(y_test_final)
        # print(len(c1_index))
        X_train = np.delete(X_train, c1_train_index, axis=0)  # 执行删除包含体动
        y_train = np.delete(y_train, c1_train_index, axis=0)  # 执行删除包含体动
        feature_test = np.delete(feature_test, c1_index, axis=0)  # 执行删除c1
        X_test = np.delete(X_test, c1_index, axis=0)  # 执行删除c1
        y_test = np.delete(y_test, c1_index, axis=0)  # 执行删除c1
        # print(len(y_test))

        # ----------C1已全部去除，开始利用分类器进行级联分类分出C2（使用ASQI1、KSQI、SNR、Sampen）----------
        c2_train_index = np.where(y_train == 5)[0]
        # 仅保留训练集中所有C级特征的均值和中位数作为最后的两个C标签，下一步进行A和B二分类
        average_c2 = np.mean(X_train[c2_train_index, :], axis=0)       # 用所有c2的平均值作为第一条C标签
        median_c2 = np. median(X_train[c2_train_index, :], axis=0)       # 用所有c2的中位数作为第二条C标签
        if len(y_test):
            c2_indx = []
            RF1 = RandomForestClassifier(n_estimators=500, random_state=42)
            RF1.fit(X_train, y_train)
            y_pred_c2 = np.round(RF1.predict(X_test))
            # RF1.fit(X_train[:, (5, 11, 13, 15)], y_train)
            # y_pred_c2 = np.round(RF1.predict(X_test[:, (5, 11, 13, 15)]))
            for num_c2 in range(0, len(y_test)):
                if y_pred_c2[num_c2] == 5:
                    y_pre_final.append(5)
                    y_test_final.append(int(y_test[num_c2]))
                    c2_indx.append(num_c2)
                    if int(y_test[num_c2]) != 5:  # 错判
                        error_index.append(num_c2)
                        error.append(feature_test[num_c2, 0])
            print(y_pre_final)
            print(y_test_final)
            X_train = np.delete(X_train, c2_train_index, axis=0)  # 执行删除c级  (若未做级联，此句可注释)
            y_train = np.delete(y_train, c2_train_index, axis=0)  # 执行删除c级  (若未做级联，此句可注释)
            feature_test = np.delete(feature_test, c2_indx, axis=0)  # 执行删除c2
            X_test = np.delete(X_test, c2_indx, axis=0)  # 执行删除c2
            y_test = np.delete(y_test, c2_indx, axis=0)  # 执行删除c2
        #
        # # 仅保留训练集中所有C级特征的均值和中位数作为最后的两个C标签，下一步进行A和B二分类
        # average_c2 = np.mean(X_train[c2_train_index, :], axis=0)       # 用所有c2的平均值作为第一条C标签
        # median_c2 = np. median(X_train[c2_train_index, :], axis=0)       # 用所有c2的中位数作为第二条C标签
        # X_train = np.delete(X_train, c2_train_index, axis=0)  # 执行删除c级
        # y_train = np.delete(y_train, c2_train_index, axis=0)  # 执行删除c级
        X_train = np.vstack((X_train, average_c2))
        X_train = np.vstack((X_train, median_c2))
        y_train = np.append(y_train, [5, 5])

        # c_test_index = np.where(y_test == 3)[0]     # 处理未被前两步识别出来的C级
        # print("经过两步C级筛选后剩余的C级索引为", c_test_index)
        # for k in range(0, len(c_test_index)):
        #     error_index.append(c_test_index[k])        # 保存其索引
        #     error.append(feature_test[c_test_index[k], 0])        # 保存其真实索引
        #     y_pre_final.append(2)
        #     y_test_final.append(3)
        #
        # feature_test = np.delete(feature_test, c_test_index, axis=0)  # 执行删除c，只剩下a和b两类
        # X_test = np.delete(X_test, c_test_index, axis=0)  # 执行删除c
        # y_test = np.delete(y_test, c_test_index, axis=0)  # 执行删除c，只剩下a和b两类
        # if 3 in y_test:
        #     print("y_test中仍然存在C级信号")
        #
        # if 3 in y_train:
        #     print("y_train中仍然存在C级信号")

        # # ------------------------先利用其余特征分出A1+A2，B1+B2（先二分类A B，再二分类1 2）------------------------------
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
            RF2 = RandomForestClassifier(n_estimators=500, random_state=42)
            # RF2.fit(X_train[:, (0, 1, 2, 3, 6, 4, 7, 8, 9, 10, 12, 16, 17, 18)], new_ytrain)
            # y_pred_b1 = np.round(RF2.predict(X_test[:, (0, 1, 2, 3, 6, 4, 7, 8, 9, 10, 12, 16, 17, 18)]))
            RF2.fit(X_train, new_ytrain)
            y_pred_b1 = np.round(RF2.predict(X_test))
            for num_b1 in range(0, len(y_test)):
                if y_pred_b1[num_b1] == 0 and X_test[num_b1, 14] == 0:
                    y_pre_final.append(1)
                    y_test_final.append(int(y_test[num_b1]))
                    if int(y_test[num_b1]) != 1:  # 错判
                        error_index.append(num_b1)
                        error.append(feature_test[num_b1, 0])
                elif y_pred_b1[num_b1] == 0 and X_test[num_b1, 14] == 1:
                    y_pre_final.append(2)
                    y_test_final.append(int(y_test[num_b1]))
                    if int(y_test[num_b1]) != 2:  # 错判
                        error_index.append(num_b1)
                        error.append(feature_test[num_b1, 0])
                elif y_pred_b1[num_b1] == 1 and X_test[num_b1, 14] == 0:
                    y_pre_final.append(3)
                    y_test_final.append(int(y_test[num_b1]))
                    if int(y_test[num_b1]) != 3:  # 错判
                        error_index.append(num_b1)
                        error.append(feature_test[num_b1, 0])
                elif y_pred_b1[num_b1] == 1 and X_test[num_b1, 14] == 1:
                    y_pre_final.append(4)
                    y_test_final.append(int(y_test[num_b1]))
                    if int(y_test[num_b1]) != 4:  # 错判
                        error_index.append(num_b1)
                        error.append(feature_test[num_b1, 0])
                else:
                    y_pre_final.append(5)
                    y_test_final.append(int(y_test[num_b1]))
                    if int(y_test[num_b1]) != 5:  # 错判
                        error_index.append(num_b1)
                        error.append(feature_test[num_b1, 0])

            print(y_pre_final)
            print(y_test_final)
            # feature_test = np.delete(feature_test, B1_index, axis=0)  # 执行删除b1
            # X_test = np.delete(X_test, B1_index, axis=0)  # 执行删除b1
            # y_test = np.delete(y_test, B1_index, axis=0)  # 执行删除b1

        # # ---------------------------分出B2---------------------------
        # if len(y_test):
        #     B2_index = []
        #     RF = RandomForestClassifier(n_estimators=500, random_state=42)
        #     RF.fit(X_train[:, (4, 7, 8, 9, 10, 12, 15, 16, 17)], y_train)
        #     y_pred_b2 = np.round(RF.predict(X_test[:, (4, 7, 8, 9, 10, 12, 15, 16, 17)]))
        #     for num_b2 in range(0, len(y_test)):
        #         if y_pred_b2[num_b2] == 2:
        #             y_pre_final.append(2)
        #             y_test_final.append(int(y_test[num_b2]))
        #             B2_index.append(num_b2)
        #             if int(y_test[num_b2]) != 2:  # 错判
        #                 error_index.append(num_b2)
        #                 error.append(feature_test[num_b2, 0])
        #     print(len(y_pre_final))
        #     print(len(y_test_final))
        #     feature_test = np.delete(feature_test, B2_index, axis=0)  # 执行删除b2
        #     X_test = np.delete(X_test, B2_index, axis=0)  # 执行删除b2
        #     y_test = np.delete(y_test, B2_index, axis=0)  # 执行删除b2
        #     print(len(y_test))
        # if len(y_test):
        #     for num_a in range(0, len(y_test)):
        #         y_pre_final.append(1)
        #         y_test_final.append(int(y_test[num_a]))
        #         if int(y_test[num_a]) != 1:  # 错判
        #             error_index.append(num_a)
        #             error.append(feature_test[num_a, 0])
        # for num in range(1,4):
        #     quality_index1 = np.where(y_pre_final == num)[0]
        #     y_pre_final = np.delete(y_pre_final, quality_index1, axis=0)
        #     quality_index2 = np.where(y_test_final == num)[0]
        #     y_test_final = np.delete(y_test_final, quality_index2, axis=0)
        print(y_pre_final)
        print(y_test_final)
        print("错判的索引为：", error_index)
        # error0 = np.sort(feature_test[error_index, 0])
        error = np.sort(error)
        # print("错判的原索引为：", error0)
        print("错判的原索引为：", error)
        print("错判的数量为：", len(error_index))
        print(np.sum((np.array(y_pre_final) - np.array(y_test_final)) == 0) / len(y_test_final))
        pre_score.append(np.sum((np.array(y_pre_final) - np.array(y_test_final)) == 0) / len(y_test_final))
        #
        # # 保存训练好的模型
        # joblib.dump(RF1, 'D:\saved_model1/' + str(file_num) + '_1.pkl')
        # joblib.dump(RF2, 'D:\saved_model1/' + str(file_num) + '_2.pkl')

        # 保存分类混淆矩阵
        plot_matrix(y_test_final, y_pre_final, [1, 2, 3, 4, 5], title='Confusion_matrix', axis_labels=['a1', 'a2', 'b1', 'b2', 'c'])
        plt.savefig(saveroot + "plot/" + str(file_num) + '.png')

        # # 输出各类别分类性能
        # class_names = ['A1', 'A2', 'B1', 'B2', 'C']
        # print("1、随机森林的性能评估：\n", classification_report(y_test_final, y_pre_final, target_names=class_names))
        # print("十折交叉验证的平均准确度为：", np.mean(pre_score))

        # 保存分类准确率
        score_result = [(file_num, np.mean(pre_score))]
        with open(os.path.join(saveroot, 'RF.txt'), 'a+') as file:
            np.savetxt(file, score_result, fmt='%.4f\t', delimiter=' ')

        # 保存分类错误的索引
        with open(os.path.join(saveroot + '\error_index/', str(file_num) + '.txt'), 'a+') as error_file:
            np.savetxt(error_file, error, fmt='%.0f\t', delimiter=' ')

        # 保存测试值与预测值
        result = list(zip(y_test_final, y_pre_final))
        with open(os.path.join(saveroot + '\\test&pre/', str(file_num) + '.txt'), 'a+') as result_file:
            np.savetxt(result_file, result, fmt='%.0f\t%.0f\t', delimiter='\t')
        # #
        # # result = list(zip(y_test, y_pre_c1, y_pred_c2, y_pred_b1, y_pred_b
        # #         # with open(r"E:\研一\信号质量评估\result4.21\test.txt", 'a+', encoding='utf-8') as file:
        # #         #     np.savetxt(file, result, fmt='%.0f\t', delimiter='\t')2))
        # # print(y_pre)
