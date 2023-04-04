import np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import  tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def cmpic(y_true, y_pred):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
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
    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

if __name__=="__main__":
    filepath = np.loadtxt('D:/bcg_data/assess_result/new/module_all.txt')   # 数据文件路径
    x = filepath[:, 0:7]
    y = filepath[:, 7]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=22)
# 训练分类器模型
clf = clf.fit(train_x, train_y)
y_pred_test = clf.predict(test_x)
y_pred_train = clf.predict(train_x)
# 计算预测准确率
score_test = np.mean(y_pred_test == test_y) * 100
score_train = np.mean(y_pred_train == train_y) * 100

class_names = ['1', '2', '3']
print("训练集的性能评估：\n", classification_report(train_y, y_pred_train, target_names=class_names))
print("测试集的性能评估：\n", classification_report(test_y, y_pred_test, target_names=class_names))
print('训练集准确率：', score_train)
print('测试集准确率：', score_test)
a = cmpic(train_y, y_pred_train)
b = cmpic(test_y, y_pred_test)
plt.show()