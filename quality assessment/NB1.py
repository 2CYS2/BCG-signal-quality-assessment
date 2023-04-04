# 使用GaussianNB分类器构建朴素贝叶斯模型
import np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]

if __name__=="__main__":
    filepath = 'D:/assess_result/new/double_all.txt'  # 数据文件路径
    # 注意数据的读取，delimiter参数是根据txt文件中的分隔符来设置的！
    dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={8: iris_type})
    X = dataset[:, 0:8]
    y = dataset[:,8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    gaussianNB = GaussianNB()
    gaussianNB.fit(X, y)

# 评估本模型在整个数据集上的表现
dataset_predict_y = gaussianNB.predict(X)
correct_predicts = (dataset_predict_y == y).sum()
accuracy = 100 * correct_predicts / y.shape[0]
print('GaussianNB, correct prediction num: {}, accuracy: {:.2f}%'.format(correct_predicts, accuracy))
