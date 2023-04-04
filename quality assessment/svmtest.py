import numpy as np
from sklearn import svm
from sklearn import model_selection

def load_data():
    # 导入数据
    data = np.loadtxt('D:/assess_result/chenzhihui/module_match2.txt', dtype=float, delimiter=None, converters={6: iris_type})
    return data

def iris_type(s):
    # 数据转为整型，数据集标签类别由string转为int
    it = {b'1': 0, b'2': 1, b'3': 2}
    return it[s]

def classifier():
    # 定义分类器
    clf = svm.SVC(C=1,  # 误差项惩罚系数
                  kernel='linear',  # 线性核 kenrel="rbf":高斯核
                  gamma = 1,
                  decision_function_shape='ovo')  # 决策函数
    return clf
# """
# 分别利用四种核函数进行训练，这些核函数都可以设置参数，例如
# decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
# decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
# 不设置的话会使用默认参数设置
# #使用linear线性核函数，C越大分类效果越好，但是可能过拟合
# clf1 = svm.SVC(C=1,kernel='linear', decision_function_shape='ovr').fit(data_train_x,data_train_y)
# #使用rbf径向基核函数,gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# clf2 = svm.SVC(C=1, kernel='rbf', gamma=1).fit(data_train_x,data_train_y)
# #使用poly多项式核函数
# clf3 = svm.SVC(kernel='poly').fit(data_train_x,data_train_y)
# #使用sigmoid神经元激活核函数
# clf4 = svm.SVC(kernel='sigmoid').fit(data_train_x,data_train_y)

def train(clf, x_train, y_train):
    # x_train：训练数据集
    # y_train：训练数据集标签
    # 训练开始
    clf.fit(x_train, y_train.ravel())  # numpy.ravel同flatten将矩阵拉平

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
    show_accuracy(clf.predict(x_train), y_train, 'traing data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')

# 训练四个特征：
data = load_data()
x, y = np.split(data, (6,), axis=1)  # x为前四列，y为第五列，x为训练数据，y为数据标签

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=42,test_size=0.3)  # 数据集划分成70%30%测试集
clf = classifier()  # 声明svm分类器对象
train(clf, x_train, y_train)  # 启动分类器进行模型训练
print_accuracy(clf, x_train, y_train, x_test, y_test)
