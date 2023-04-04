import np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]

if __name__=="__main__":
    filepath = 'D:/assess_result/chenzhihui/module_match2.txt'  # 数据文件路径
    # 注意数据的读取，delimiter参数是根据txt文件中的分隔符来设置的！
    dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={6: iris_type})
    iris_feature = dataset[:, 0:6]
    iris_target = dataset[:, 6]

print (iris_target)
#scikit-learn 已经将花的原名称进行了转换，其中 0, 1, 2 分别代表 Iris Setosa, Iris Versicolour 和 Iris Virginica
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.3,random_state=42)
print("--------------------------------------------------------------------------------------")
print(target_train)

dt_model = DecisionTreeClassifier(max_depth=8)
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)
scores = dt_model.score(feature_test, target_test)

print(predict_results)
print(target_test)
print('准确率为：',accuracy_score(predict_results, target_test))
