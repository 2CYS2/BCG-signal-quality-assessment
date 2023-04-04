import np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]

if __name__=="__main__":
    filepath = 'D:/assess_result/chenzhihui/module_match2.txt'  # 数据文件路径
    # 注意数据的读取，delimiter参数是根据txt文件中的分隔符来设置的！
    dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={6: iris_type})
    x = dataset[:, 0:6]
    y = dataset[:, 6]


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier(max_depth=8)
# 训练分类器模型
clf = clf.fit(train_x, train_y)
y_predicted = clf.predict(test_x)
# 计算预测准确率
score = np.mean(y_predicted == test_y) * 100

# dt_model = DecisionTreeClassifier()
# dt_model.fit(train_x, train_y)
# predict_y = dt_model.predict(test_x)
# score = dt_model.score(test_x, test_y)

print(y_predicted)
print(test_y)
print('准确率：', score)