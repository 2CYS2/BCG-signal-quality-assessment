"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: stacking_test.py
 @Function: 将多个集成学习器堆叠为一个新的分类器
 @DateTime: 2023/3/28 19:48
 @SoftWare: PyCharm
"""
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import lightgbm as lgb

from xgboost import XGBClassifier

if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\result3.21\delete_nan_inf3.28\norm_30s/"
    label = ('RF', 'DT', 'KNN', 'XG', 'Lgb', 'Cb')
    filenames = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总\2.22更新\result2\余老师绘图\file_num.txt", encoding='utf-8')
    filenames = [955, 560]
    for filename in filenames:
        file_num = int(filename)
        filename = str(int(filename))
        feature_file = np.loadtxt(os.path.join(filepath, filename + '.txt'), encoding='utf-8')
        X = feature_file[:, 1:17]
        y = feature_file[:, 17]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2

        # # strKFold = ShuffleSplit(train_size=.5,test_size=.4,n_splits=10)   # 打乱后随机分配
        strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        kfold = strtfdKFold.split(X, y)
        for k, (train, test) in enumerate(kfold):
            print("------------------第%d折----------------" % (k+1))
            RF = RandomForestClassifier(n_estimators=500, random_state=42)
            clf = tree.DecisionTreeClassifier(max_depth=22)
            knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors=1表示k=1
            xg = XGBClassifier(eval_metric='mlogloss')
            LGBM = lgb.LGBMClassifier(num_leaves=20, learning_rate=0.05, n_estimators=20)
            cb = CatBoostClassifier(iterations=12, learning_rate=0.05, depth=10, verbose=False)
            Ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=35, min_samples_split=30, min_samples_leaf=5), \
                                     algorithm="SAMME", n_estimators=35, learning_rate=0.1, random_state=42)

            svclf = svm.SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
            treeclf = DecisionTreeClassifier()
            gbdtclf = GradientBoostingClassifier(learning_rate=0.7)
            lrclf = LogisticRegression()

            scclf = StackingCVClassifier(classifiers=[xg, Ada, LGBM, RF], meta_classifier=clf, cv=5)
            scclf.fit(X[train], y[train])
            scclf_pre = scclf.predict(X[test])
            print('准确度：', accuracy_score(scclf_pre, y[test]))