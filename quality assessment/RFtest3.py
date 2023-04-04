import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]


filepath = r'E:\研一\信号质量评估\Result\result_10/221.txt'  # 数据文件路径
dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={12: iris_type})
#dataset = np.loadtxt(filepath)
X = dataset[:, 1:7]
y = dataset[:, 12]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#实例化一个随机森林分类器对象   指定包含5棵决策树，最大叶子节点数为16，用5个线程进行训练
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=5)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
print('accuracy:',accuracy_score(y_test, y_pred_rf))

# 计算特征重要性
# iris = load_iris()
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rnd_clf.fit(iris["data"], iris['target'])
# for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
#     print(name, score)
