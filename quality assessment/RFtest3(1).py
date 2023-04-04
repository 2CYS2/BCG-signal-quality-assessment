import np
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]


filepath = 'D:/assess_result/new/module_all.txt'  # 数据文件路径
dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={7: iris_type})
#dataset = np.loadtxt(filepath)
X = dataset[:, 0:7]
y = dataset[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#实例化一个随机森林分类器对象   指定包含5棵决策树，最大叶子节点数为16，用5个线程进行训练
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=8, n_jobs=5)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
print('accuracy:',accuracy_score(y_test, y_pred_rf))

# 计算特征重要性
# iris = load_iris()
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rnd_clf.fit(iris["data"], iris['target'])
# for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
#     print(name, score)
def cmpic(y_true, y_pred):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
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
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,8), dpi=120)
    #set the fontsize of label.
    #for label in plt.gca().xaxis.get_ticklabels():
    #    label.set_fontsize(8)
    #text portion
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    #offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

a = cmpic(y_test, y_pred_rf)
pyplot.show()