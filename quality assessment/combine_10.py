from matplotlib import pyplot
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, ShuffleSplit, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier


def importance(name):
    feature_names = ['BSQI1', 'BSQI2', 'ISQI1', 'ISQI2', 'RsdSQI', 'ASQI1', 'ASQI2', 'PSQI', 'CSQI', 'Avg_corr',
                     'Rsd_corr', 'SNR', 'SSQI', 'KSQI', 'artiafct', 'Sampen', 'MSQI', 'PURSQI', 'AJH', 'AJL']
    perm_importance = permutation_importance(name, X[test], y[test])
    # sorted_idx = perm_importance.importances_mean.argsort()
    # plt.barh(np.array(feature_names)[sorted_idx], np.array(perm_importance.importances_mean)[sorted_idx])
    # print(perm_importance.importances_mean)
    sorted_idx = perm_importance.importances_mean.argsort()     # 从小到大排序
    plt.barh(np.array(feature_names), np.array(perm_importance.importances_mean))

    return perm_importance.importances_mean


#混淆矩阵
def cmpic(y_true, y_pred, i):
        def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['A','B','C']):
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
        plt.gcf().subplots_adjust(bottom=0.05,hspace=0.5)
        if (i == 1):
            plot_confusion_matrix (cm_normalized, title='Normalized confusion matrix: RF')
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
    regressor = RandomForestClassifier(n_estimators=500, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred1 = np.round(regressor.predict(X_test))
    score1 = np.mean(y_pred1 == y_test)
    print ("1、随机森林分类准确率为: %.2f%%" % (score1 * 100.0))

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
    score3 = accuracy_score(y_test,y_pred3)
    print("3、k近邻模型分类准确率为: %.2f%%" % (score3 * 100.0))

    # 4、XGboost模型
    model = XGBClassifier(eval_metric='auc')
    model.fit(X_train, y_train)
    y_pred4 = model.predict(X_test)
    score4 = accuracy_score(y_test, y_pred4)
    print("4、XGboost模型分类准确率: %.2f%%" % (score4 * 100.0))
    #各分类器性能
    # class_names = ['A', 'B', 'C']
    # print("1、随机森林的性能评估：\n", classification_report(y_test, y_pred1, target_names=class_names))
    # print("2、决策树的性能评估：\n", classification_report(y_test, y_pred2, target_names=class_names))
    # print("3、K近邻的性能评估：\n", classification_report(y_test, y_pred3, target_names=class_names))
    # print("4、KXGboost的性能评估：\n", classification_report(y_test, y_pred4, target_names=class_names))
    if show_confusion:
        plt.figure(figsize=(8,8))
        plt.subplot(3, 2, 1)
        a = cmpic(y_test, y_pred1,1)

        plt.subplot(3, 2, 2)
        b = cmpic(y_test, y_pred2,2)

        plt.subplot(3, 2, 3)
        c = cmpic(y_test, y_pred3,3)

        plt.subplot(3, 2, 4)
        c = cmpic(y_test, y_pred3,4)

        plt.subplot(3, 2, 5)
        importance(regressor)
        plt.xlabel("Permutation Importance: RF")

        plt.subplot(3, 2, 6)
        importance(clf)
        plt.xlabel("Permutation Importance: DT")
        pyplot.show()


if __name__=="__main__":
    # 数据文件路径
    filepath = r"E:\研一\信号质量评估\result11.8\normalization\10s/"
    saveroot = r"E:\研一\信号质量评估\result11.8\normalization\个体内训练分类(获取特征重要性)/"
    # filepath2 = np.loadtxt(r'E:\研一\信号质量评估\Result_new\result_norm_10/B.txt')
    # filepath3 = np.loadtxt(r'E:\研一\信号质量评估\Result_new\result_norm_10/C.txt')
    # filepath = np.vstack((filepath1,filepath2,filepath3))
    label = ('RF', 'DT', 'KNN', 'XG', 'Lgb', 'Cb')
    # filenames = os.listdir(filepath)  # 获取该文件夹中所有样本的特征结果
    filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num_new.txt", encoding='utf-8')
    # filenames = [955]
    # filenames = filenames[35:]
    for filename in filenames:
        file_num = int(filename)
        filename = str(int(filename))
        print("正在处理：", filename + '.txt')
        feature_file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
        X = feature_file[:, 1:21]
        y = feature_file[:, 21]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        print("%s的结果为：" % filename)
        # strKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 分层交叉验证
        # # strKFold = ShuffleSplit(train_size=.5,test_size=.4,n_splits=10)   # 打乱后随机分配
        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        kfold = strtfdKFold.split(X, y)
        # 1、随机森林模型
        RF = RandomForestClassifier(n_estimators=500, random_state=42)
        # scores1 = cross_val_score(RF, X, y, cv=strKFold)
        # print("Cross validation scores:{}".format(scores1))
        # print("Mean cross validation score:{:2f}".format(scores1.mean()))
        scores_rf, scores_dt, scores_knn, scores_xg, scores_lgb, scores_cb, scores_ada = [], [], [], [], [], [], []
        import_rf, import_dt, import_knn, import_xg, import_lgb, import_cb, import_ada= [], [], [], [], [], [], []
        for k, (train, test) in enumerate(kfold):
            # print(train)
            print("------------------第 %.0f 折-----------------" % (k+1))
            print("1、随机森林模型")
            RF.fit(X[train], y[train])
            score = RF.score(X[test], y[test])
            scores_rf.append(score)
            import_rf.append(importance(RF))
            # print(score)
        # RF.fit(X_train, y_train)
        #     y_pred1 = np.round(RF.predict(X[test]))
        # score1 = np.mean(y_pred1 == y_test)
        # print ("1、随机森林分类准确率为: %.2f%%" % (score1 * 100.0))

        # # 2、决策树模型
        #     print("2、决策树模型")
        #     clf = tree.DecisionTreeClassifier(max_depth=22)
        #     # scores2 = cross_val_score(clf, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores2))
        #     # print("Mean cross validation score:{:2f}".format(scores2.mean()))
        #     # for k, (train, test) in enumerate(kfold):
        #     clf.fit(X[train], y[train])
        #     score = clf.score(X[test], y[test])
        #     scores_dt.append(score)
        #     # import_dt.append(importance(clf))
        #     # print(score)
        #
        # # clf = clf.fit(X_train, y_train)
        # # y_pred2 = clf.predict(X_test)
        # # score2 = np.mean(y_pred2 == y_test)
        # # print("2、决策树模型分类准确率为：%.2f%%" % (score2 * 100.0))
        #
        # # 3、K近邻模型
        #     print("3、K近邻模型")
        #     knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors=1表示k=1
        #     # scores3 = cross_val_score(knn, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores3))
        #     # print("Mean cross validation score:{:2f}".format(scores3.mean()))
        # # for k, (train, test) in enumerate(kfold):
        #     knn.fit(X[train], y[train])
        #     score = knn.score(X[test], y[test])
        #     scores_knn.append(score)
        #     # import_knn.append(importance(knn))
        #     # print(score)
        #
        # # knn = knn.fit(X_train, y_train)
        # # y_pred3 = knn.predict(X_test)
        # # score3 = accuracy_score(y_test,y_pred3)
        # # print("3、k近邻模型分类准确率为: %.2f%%" % (score3 * 100.0))

        # # 4、XGboost模型
        #     print("4、XGboost模型")
        #     xg = XGBClassifier(eval_metric='mlogloss')
        #     # scores4 = cross_val_score(xg, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores4))
        #     # print("Mean cross validation score:{:2f}".format(scores4.mean()))
        # # for k, (train, test) in enumerate(kfold):
        #     xg.fit(X[train], y[train])
        #     score = xg.score(X[test], y[test])
        #     scores_xg.append(score)
        #     import_xg.append(importance(xg))
        #     # print(score)

        # xg.fit(X_train, y_train)
        # y_pred4 = xg.predict(X_test)
        # score4 = accuracy_score(y_test, y_pred4)
        # print("4、XGboost模型分类准确率: %.2f%%" % (score4 * 100.0))

        # 5、LightGBM模型
        #     print("5、LightGBM模型")
        #     LGBM = lgb.LGBMClassifier(num_leaves=20, learning_rate=0.05, n_estimators=20)
        #     # scores5 = cross_val_score(LGBM, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores5))
        #     # print("Mean cross validation score:{:2f}".format(scores5.mean()))
        # # for k, (train, test) in enumerate(kfold):
        #     LGBM.fit(X[train], y[train])
        #     score = LGBM.score(X[test], y[test])
        #     scores_lgb.append(score)
        #     import_lgb.append(importance(LGBM))
            # print(score)

        # LGBM.fit(X_train, y_train)
        # y_pred5 = LGBM.predict(X_test)
        # score5 = accuracy_score(y_test, y_pred5)
        # print('5、LightGBM模型分类准确率: %.2f%%' % (score5*100))

        # # 6、CatBoost模型
        #     print("6、CatBoost模型")
        #     cb = CatBoostClassifier(iterations=12, learning_rate=0.05, depth=10, verbose=False)
        #     # scores6 = cross_val_score(cb, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores6))
        #     # print("Mean cross validation score:{:2f}".format(scores6.mean()))
        # # for k, (train, test) in enumerate(kfold):
        #     cb.fit(X[train], y[train])
        #     score = cb.score(X[test], y[test])
        #     scores_cb.append(score)
        #     # import_cb.append(importance(cb))
        #     # print(score)
        # #
        # # 7、AdaBoost模型
        #     print("7、AdaBoost模型")
        #     Ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=35, min_samples_split=30, min_samples_leaf=5), \
        #                              algorithm="SAMME", n_estimators=35, learning_rate=0.1, random_state=42)
        #     # scores6 = cross_val_score(cb, X, y, cv=strKFold)
        #     # print("Cross validation scores:{}".format(scores6))
        #     # print("Mean cross validation score:{:2f}".format(scores6.mean()))
        # # for k, (train, test) in enumerate(kfold):
        #     Ada.fit(X[train], y[train])
        #     score = Ada.score(X[test], y[test])
        #     scores_ada.append(score)
        #     import_ada.append(importance(Ada))
        #     # print(score)

        # cb.fit(X_train, y_train, verbose=False)
        # y_pred6 = cb.predict(X_test)
        # score6 = accuracy_score(y_test, y_pred6)
        # print('6、CatBoost模型分类准确率: %.2f%%' % (score6*100))

        print('平均准确率为: %.3f' %(np.mean(scores_rf)))
        # print('平均准确率为: %.3f' %(np.mean(scores_dt)))
        # print('平均准确率为: %.3f' %(np.mean(scores_knn)))
        # print('平均准确率为: %.3f' %(np.mean(scores_xg)))
        # print('平均准确率为: %.3f' %(np.mean(scores_lgb)))
        # # print('平均准确率为: %.3f' %(np.mean(scores_cb)))
        # print('平均准确率为: %.3f' %(np.mean(scores_ada)))
        print('各折运算的准确率为:', scores_rf)
        # print('准确率为:', scores_dt)
        # print('准确率为:', scores_knn)
        # print('准确率为:', scores_xg)
        # print('准确率为:', scores_lgb)
        # # print('准确率为:', scores_cb)
        # print('准确率为:', scores_ada)
        # score_result = [(file_num, np.mean(scores_rf), np.mean(scores_dt), np.mean(scores_knn), np.mean(scores_xg), np.mean(scores_lgb), np.mean(scores_cb), np.mean(scores_ada))]
        score_result = [(file_num, np.mean(scores_rf))]
        with open(os.path.join(saveroot, 'score.txt'), 'a+') as file:
            np.savetxt(file, score_result, fmt='%.4f\t', delimiter=' ')

        model_name = ['RF', 'DT', 'KNN', 'XGboost', 'LightGBM', 'Catboost', 'Adaboost']
        # model_name = ['RF', 'XGboost', 'LightGBM', 'Adaboost']
        model = []
        model.append(import_rf)
        # model.append(import_dt)
        # model.append(import_knn)
        # model.append(import_xg)
        # model.append(import_lgb)
        # # model.append(import_cb)
        # model.append(import_ada)
        with open(os.path.join(saveroot, model_name[0] + '.txt'), 'a+') as file1:
            np.savetxt(file1, np.mean(list(zip(import_rf)), axis=0), fmt='%.4f\t', delimiter=' ')

        # with open(os.path.join(saveroot, model_name[1] + '.txt'), 'a+') as file2:
        #     np.savetxt(file2, np.mean(list(zip(import_dt)), axis=0), fmt='%.4f\t', delimiter=' ')
        #
        # with open(os.path.join(saveroot, model_name[2] + '.txt'), 'a+') as file3:
        #     np.savetxt(file3, np.mean(list(zip(import_knn)), axis=0), fmt='%.4f\t', delimiter=' ')

        # with open(os.path.join(saveroot, model_name[3] + '.txt'), 'a+') as file4:
        #     np.savetxt(file4, np.mean(list(zip(import_xg)), axis=0), fmt='%.4f\t', delimiter=' ')
        #
        # with open(os.path.join(saveroot, model_name[4] + '.txt'), 'a+') as file5:
        #     np.savetxt(file5, np.mean(list(zip(import_lgb)), axis=0), fmt='%.4f\t', delimiter=' ')

        # with open(os.path.join(saveroot, model_name[5] + '.txt'), 'a+') as file6:
        #     np.savetxt(file6, np.mean(list(zip(import_cb)), axis=0), fmt='%.4f\t', delimiter=' ')

        # with open(os.path.join(saveroot, model_name[6] + '.txt'), 'a+') as file7:
        #     np.savetxt(file7, np.mean(list(zip(import_ada)), axis=0), fmt='%.4f\t', delimiter=' ')



        # result_all = []
        # result_all = result(filename, y_test, y_pred1, y_pred2, y_pred3, y_pred4)
        #
        # print(result_all)
        # with open(os.path.join(outputpath, filename), 'a+') as f:
        #         np.savetxt(f, result_all, fmt='%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t', delimiter='\t')

        # #各分类器性能
        # class_names = ['A', 'B', 'C']
        # print("1、随机森林的性能评估：\n", classification_report(y[test], y_pred1, target_names=class_names))
        # # print("2、决策树的性能评估：\n", classification_report(y_test, y_pred2, target_names=class_names))
        # # print("3、K近邻的性能评估：\n", classification_report(y_test, y_pred3, target_names=class_names))
        # # print("4、KXGboost的性能评估：\n", classification_report(y_test, y_pred4, target_names=class_names))
        #
        # plt.figure()
        # a = cmpic(y[test], y_pred1, 1)

        # y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
        # model = [RF, clf, knn, xg, LGBM, cb]
        # model_name = ['RF', 'DT', 'KNN', 'XGboost', 'LightGBM', 'Catboost']
        # #
        # fig1 = plt.figure(figsize=(15,8))
        # # plt.rcParams.update({'font.size': 20})
        # for i in range(0, 6):
        #     plt.subplot(2, 3, i+1)
        #     a = cmpic(y_test, y_pred[i], i+1)
        # #
        # fig2 = plt.figure(figsize=(8,8))
        # for j in range(0, 6):
        #     plt.subplot(3, 2, j+1)
        #     importance(model[j])
        #     plt.xlabel("Permutation Importance: %s" % model_name[j])
        # plt.show()