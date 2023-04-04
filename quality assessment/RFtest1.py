#随机森林分类
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def iris_type(s):
    class_label={b'1':1,b'2':2,b'3':3}
    return class_label[s]

if __name__=="__main__":
    filepath = 'D:/assess_result/zhangmin/module_match.txt'  # 数据文件路径
    # 注意数据的读取，delimiter参数是根据txt文件中的分隔符来设置的！
    dataset = np.loadtxt(filepath, dtype=float, delimiter=None, converters={6: iris_type})
    #dataset = np.loadtxt(filepath)
    X = dataset[:, 0:6]
    y = dataset[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    regressor = RandomForestRegressor(n_estimators=500, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = np.round(regressor.predict(X_test))
    accuracy = np.mean(y_pred == y_test)
    print ("y_test\n",y_test)
    print ("y_pred\n",y_pred)
    print ("accuracy:",accuracy)
    # plot
    pyplot.bar(range(len(regressor.feature_importances_)), regressor.feature_importances_)
    pyplot.show()