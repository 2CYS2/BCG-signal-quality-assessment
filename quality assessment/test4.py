"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: test4.py
 @Function: 按个体患病严重程度划分训练测试
 @DateTime: 2023/3/26 15:46
 @SoftWare: PyCharm
"""
"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: test3.py
 @Function: 
 @DateTime: 2023/3/15 9:42
 @SoftWare: PyCharm
"""
from matplotlib import pyplot
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
from random import sample
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier


def importance(name):
    feature_names = ['BSQI1', 'BSQI2', 'ISQI1', 'ISQI2', 'RsdSQI', 'ASQI','PSQI', 'CSQI', 'Avg_corr', 'SNR', 'SSQI', 'KSQI', 'artiafct']
    perm_importance = permutation_importance(name, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(np.array(feature_names)[sorted_idx], np.array(perm_importance.importances_mean)[sorted_idx])


def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['A', 'B', 'C']):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 混淆矩阵
def cmpic(y_true, y_pred, i):

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


def test(parameter, machine1, machine2, machine3, machine4, label1, label2):
    parameter.append(machine1[label1][label2])
    parameter.append(machine2[label1][label2])
    parameter.append(machine3[label1][label2])
    parameter.append(machine4[label1][label2])

    return parameter


def result(filename, y_test, y_pred1, y_pred2, y_pred3, y_pred4):
    file, acc, macro_pre, macro_recall, macro_f1, weight_pre, weight_recall, weight_f1, support = [], [], [], [], [], [], [], [], []
    class_names = ['A', 'B', 'C']

    machine1 = classification_report(y_test, y_pred1, target_names=class_names, output_dict=True)
    machine2 = classification_report(y_test, y_pred2, target_names=class_names, output_dict=True)
    machine3 = classification_report(y_test, y_pred3, target_names=class_names, output_dict=True)
    machine4 = classification_report(y_test, y_pred4, target_names=class_names, output_dict=True)

    label1 = ('macro avg', 'weighted avg')
    label2 = ('precision', 'recall', 'f1-score', 'support')
    file.append(filename)
    file.append(filename)
    file.append(filename)
    file.append(filename)
    acc.append(machine1['accuracy'])
    acc.append(machine2['accuracy'])
    acc.append(machine3['accuracy'])
    acc.append(machine4['accuracy'])
    macro_pre = test(macro_pre, machine1, machine2, machine3, machine4, label1[0], label2[0])
    macro_recall = test(macro_recall, machine1, machine2, machine3, machine4, label1[0], label2[1])
    macro_f1 = test(macro_f1, machine1, machine2, machine3, machine4, label1[0], label2[2])
    support = test(support, machine1, machine2, machine3, machine4, label1[0], label2[3])

    weight_pre = test(weight_pre, machine1, machine2, machine3, machine4, label1[1], label2[0])
    weight_recall = test(weight_recall, machine1, machine2, machine3, machine4, label1[1], label2[1])
    weight_f1 = test(weight_f1, machine1, machine2, machine3, machine4, label1[1], label2[2])

    print("1、随机森林的性能评估：\n", classification_report(y_test, y_pred1, target_names=class_names))
    print("2、决策树的性能评估：\n", classification_report(y_test, y_pred2, target_names=class_names))
    print("3、K近邻的性能评估：\n", classification_report(y_test, y_pred3, target_names=class_names))
    print("4、KXGboost的性能评估：\n", classification_report(y_test, y_pred4, target_names=class_names))

    result = list(zip(acc, macro_pre, macro_recall, macro_f1, weight_pre, weight_recall, weight_f1, support))

    return result


def model_construct(X_train, X_test, y_train, y_test, show_confusion):
    """
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param show_confusion:
    :return:
    """
    # 1、随机森林模型
    regressor = RandomForestRegressor(n_estimators=500, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred1 = np.round(regressor.predict(X_test))
    score1 = np.mean(y_pred1 == y_test)
    print("1、随机森林分类准确率为: %.2f%%" % (score1 * 100.0))

    # 2、决策树模型
    clf = tree.DecisionTreeClassifier(max_depth=22)
    clf = clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)
    score2 = np.mean(y_pred2 == y_test)
    print("2、决策树模型分类准确率为：%.2f%%" % (score2 * 100.0))

    # 3、K近邻模型
    knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors=1表示k=1
    knn = knn.fit(X_train, y_train)
    y_pred3 = knn.predict(X_test)
    score3 = accuracy_score(y_test, y_pred3)
    print("3、k近邻模型分类准确率为: %.2f%%" % (score3 * 100.0))

    # 4、XGboost模型
    model = XGBClassifier(eval_metric='auc')
    model.fit(X_train, y_train)
    y_pred4 = model.predict(X_test)
    score4 = accuracy_score(y_test, y_pred4)
    print("4、XGboost模型分类准确率: %.2f%%" % (score4 * 100.0))
    # 各分类器性能
    # class_names = ['A', 'B', 'C']
    # print("1、随机森林的性能评估：\n", classification_report(y_test, y_pred1, target_names=class_names))
    # print("2、决策树的性能评估：\n", classification_report(y_test, y_pred2, target_names=class_names))
    # print("3、K近邻的性能评估：\n", classification_report(y_test, y_pred3, target_names=class_names))
    # print("4、KXGboost的性能评估：\n", classification_report(y_test, y_pred4, target_names=class_names))
    if show_confusion:
        plt.figure(figsize=(8, 8))
        plt.subplot(3, 2, 1)
        a = cmpic(y_test, y_pred1, 1)

        plt.subplot(3, 2, 2)
        b = cmpic(y_test, y_pred2, 2)

        plt.subplot(3, 2, 3)
        c = cmpic(y_test, y_pred3, 3)

        plt.subplot(3, 2, 4)
        c = cmpic(y_test, y_pred3, 4)

        plt.subplot(3, 2, 5)
        importance(regressor)
        plt.xlabel("Permutation Importance: RF")

        plt.subplot(3, 2, 6)
        importance(clf)
        plt.xlabel("Permutation Importance: DT")
        pyplot.show()


if __name__ == "__main__":
    # 数据文件路径
    filepath =  r"E:\研一\信号质量评估\result4.9\normalization\10s/"
    saveroot = r"E:\研一\信号质量评估\result4.9\normalization\留一法\10s/"
    # filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')
    label = ('RF', 'DT', 'KNN', 'XG', 'Lgb', 'Cb')

    # filenames = os.listdir(filepath)
    # filenames_train = sample(filenames, 36)
    #
    # for i in filenames_train:
    #     filenames.remove(i)
    # filenames_test = sample(filenames, 4)
    #
    # for j in filenames_test:
    #     filenames.remove(j)
    #
    # filenames_verify = filenames

    # filenames_test = ['971.txt', '935.txt', '966.txt', '541.txt']
    acc_result = []
    scores_rf, scores_dt, scores_knn, scores_xg, scores_lgb, scores_cb = [], [], [], [], [], []
    # for k in range(0, 5):   # 按患病程度分层抽样交叉验证
    #     filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8').tolist()
    #     print("------------------第 %.0f 折-----------------" % (k + 1))
    #     filename1 = sample(list(filenames[5:19]), 1)
    #     filename2 = sample(list(filenames[19:27]), 1)
    #     filename3 = sample(list(filenames[27:40]), 1)
    #     # print(filename3)
    #     filenames_test = [filenames[k], filename1[0], filename2[0], filename3[0]]
    #     # filenames_test = [filenames[k], filename1[0], filename1[1], filename1[2], filename2[0], filename2[1], filename3[0], filename3[1], filename3[2]]
    #     print(filenames_test)
    #     for i in filenames_test:
    #         filenames.remove(i)
    #     filenames_train = filenames
    #
    #     feature_train = np.loadtxt(os.path.join(filepath, str(int(filenames_train[0]))+'.txt'), encoding='utf-8')
    #     for f1 in range(1, 36):
    #         # print(filenames_train[f1])
    #         feature_file = np.loadtxt(os.path.join(filepath, str(int(filenames_train[f1]))+'.txt'), encoding='utf-8')
    #         feature_train = np.vstack((feature_train, feature_file))
    #     print(np.shape(feature_train))
    #     # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\train.txt", feature_train, fmt='%.4f\t')
    #
    #     feature_test = np.loadtxt(os.path.join(filepath, str(int(filenames_test[0]))+'.txt'), encoding='utf-8')
    #     for f2 in range(1, 4):
    #         # print(filenames_test[f2])
    #         feature_file = np.loadtxt(os.path.join(filepath, str(int(filenames_test[f2]))+'.txt'), encoding='utf-8')
    #         feature_test = np.vstack((feature_test, feature_file))
    #     print(np.shape(feature_test))
    # #
    for k in range(0, 1):     # 按顺序十折交叉验证
        print("------------------第 %.0f 折-----------------" % (k + 1))
        filenames = os.listdir(filepath)
        # filenames_test = filenames[k:k + 1]
        # filenames_test = ['971.txt', '935.txt', '966.txt', '541.txt']
        filenames_test = ['971.txt']
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
        #
        # np.savetxt(r"E:\研一\信号质量评估\result3.13\delete_nan_inf\test.txt", feature_test, fmt='%.4f\t')
        #
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

        X_train = feature_train[:, (1,2,6,7,8,9,10,11,12,13,14,15,16,19,20)]
        y_train = feature_train[:, 21]
        X_test = feature_test[:, (1,2,6,7,8,9,10,11,12,13,14,15,16,19,20)]
        y_test = feature_test[:, 21]
        # X_verify = feature_verify[:, 1:14]
        # y_verify = feature_verify[:, 14]
        # print("%s的结果为：" % filename)
        # 1、随机森林模型
        RF = RandomForestClassifier(n_estimators=500, random_state=42)
        RF.fit(X_train, y_train)
        # y_pred1_verify = np.round(regressor.predict(X_verify))
        # score1_verify = np.mean(y_pred1_verify == y_verify)
        y_pred1_test = np.round(RF.predict(X_test))
        score1_test = np.mean(y_pred1_test == y_test)
        # print("1、随机森林验证集准确率为: %.2f%%" % (score1_verify * 100.0))
        print("1、随机森林测试集准确率为: %.2f%%" % (score1_test * 100.0))

        # # 2、决策树模型
        # clf = tree.DecisionTreeClassifier(max_depth=22)
        # clf = clf.fit(X_train, y_train)
        # # y_pred2_verify = clf.predict(X_verify)
        # # score2_verify = np.mean(y_pred2_verify == y_verify)
        # y_pred2_test = clf.predict(X_test)
        # score2_test = np.mean(y_pred2_test == y_test)
        # # print("2、决策树模型验证集准确率为：%.2f%%" % (score2_verify * 100.0))
        # print("2、决策树模型测试集准确率为：%.2f%%" % (score2_test * 100.0))
        #
        # # 3、K近邻模型
        # knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors=1表示k=1
        # knn = knn.fit(X_train, y_train)
        # # y_pred3_verify = knn.predict(X_verify)
        # # score3_verify = accuracy_score(y_verify, y_pred3_verify)
        # y_pred3_test = knn.predict(X_test)
        # score3_test = accuracy_score(y_test, y_pred3_test)
        # # print("3、k近邻模型验证集准确率为: %.2f%%" % (score3_verify * 100.0))
        # print("3、k近邻模型测试集准确率为: %.2f%%" % (score3_test * 100.0))
        # #
        # # 4、XGboost模型
        # xg = XGBClassifier(eval_metric='mlogloss')
        # xg.fit(X_train, y_train)
        # # y_pred4_verify = model.predict(X_verify)
        # # score4_verify = accuracy_score(y_verify, y_pred4_verify)
        # y_pred4_test = xg.predict(X_test)
        # score4_test = accuracy_score(y_test, y_pred4_test)
        # # print("4、XGboost模型验证集准确率: %.2f%%" % (score4_verify * 100.0))
        # print("4、XGboost模型测试集准确率: %.2f%%" % (score4_test * 100.0))
        #
        # # 5、LightGBM模型
        # LGBM = lgb.LGBMClassifier(num_leaves=20, learning_rate=0.05, n_estimators=20)
        # LGBM.fit(X_train, y_train)
        # # y_pred5_verify = model.predict(X_verify)
        # # score5_verify = accuracy_score(y_verify, y_pred5_verify)
        # y_pred5_test = LGBM.predict(X_test)
        # score5_test = accuracy_score(y_test, y_pred5_test)
        # # print('5、LightGBM模型验证集准确率: %.2f%%' % (score5_verify * 100))
        # print('5、LightGBM模型测试集准确率: %.2f%%' % (score5_test * 100))
        # #
        # # 6、CatBoost模型
        # cb = CatBoostClassifier(iterations=12, learning_rate=0.05, depth=10)
        # cb.fit(X_train, y_train, verbose=False)
        # # y_pred6_verify = cb.predict(X_verify)
        # # score6_verify = accuracy_score(y_verify, y_pred6_verify)
        # y_pred6_test = cb.predict(X_test)
        # score6_test = accuracy_score(y_test, y_pred6_test)
        # # print('6、CatBoost模型验证集准确率: %.2f%%' % (score6_verify * 100))
        # print('6、CatBoost模型测试集准确率: %.2f%%' % (score6_test * 100))
        #
        # # 7、AdaBoost模型
        # print("7、AdaBoost模型")
        # Ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=35, min_samples_split=30, min_samples_leaf=5), \
        #                          algorithm="SAMME", n_estimators=35, learning_rate=0.1, random_state=42)
        # # scores6 = cross_val_score(cb, X, y, cv=strKFold)
        # # print("Cross validation scores:{}".format(scores6))
        # # print("Mean cross validation score:{:2f}".format(scores6.mean()))
        # # for k, (train, test) in enumerate(kfold):
        # Ada.fit(X_train, y_train)
        # score7_test = Ada.score(X_test, y_test)
        # print('7、AdaBoost模型模型测试集准确率: %.2f%%' % (score7_test * 100))
        # print(score)

        # score_result = [(k, score1_test, score2_test, score3_test, score4_test, score5_test, score6_test, score7_test)]
        # score_result = [(k, score1_test, score4_test, score5_test, score7_test)]
        # with open(os.path.join(saveroot, 'feature_cut.txt'), 'a+') as file:
        #     np.savetxt(file, score_result, fmt='%.4f\t', delimiter=' ')

        # acc_result.append(score5_test * 100.0)
    # print(acc_result)
        # # #
        # result_all = []
        # result_all = result(filename, y_test, y_pred1, y_pred2, y_pred3, y_pred4)
        #
        # print(result_all)
        # with open(os.path.join(outputpath, filename), 'a+') as f:
        #         np.savetxt(f, result_all, fmt='%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t', delimiter='\t')

        # 各分类器性能
        class_names = ['A', 'B', 'C']
        print("1、随机森林的性能评估：\n", classification_report(y_test, y_pred1_test, target_names=class_names))
        # print("2、决策树的性能评估：\n", classification_report(y_test, y_pred2_test, target_names=class_names))
        # print("3、K近邻的性能评估：\n", classification_report(y_test, y_pred3_test, target_names=class_names))
        # print("4、KXGboost的性能评估：\n", classification_report(y_test, y_pred4_test, target_names=class_names))


        plt.figure()
        a = cmpic(y_test, y_pred1_test, 1)


        # y_pred = [y_pred1_test, y_pred2_test, y_pred3_test, y_pred4_test, y_pred5_test, y_pred6_test]
        # model = [RF, clf, knn, xg, LGBM, cb]
        # model_name = ['RF', 'DT', 'KNN', 'XGboost', 'LightGBM', 'Catboost']
        #
        # fig1 = plt.figure(figsize=(15, 8))
        # # plt.rcParams.update({'font.size': 20})
        # for i in range(0, 6):
        #     plt.subplot(2, 3, i + 1)
        #     a = cmpic(y_test, y_pred[i], i + 1)
        #
        # fig2 = plt.figure(figsize=(8, 8))
        # for j in range(0, 6):
        #     plt.subplot(3, 2, j + 1)
        #     importance(model[j])
        #     plt.xlabel("Permutation Importance: %s" % model_name[j])
        plt.show()
