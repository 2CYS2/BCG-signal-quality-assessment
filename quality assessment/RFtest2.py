import numpy as np

# 导入数据，路径中要么用\\或/或者在路径前加r
dataset = np.loadtxt(r'D:/assess_result/yukaibing/module_match.txt')

# 输出数据预览
#print(dataset.head())

# 准备训练数据
# 自变量：汽油税、人均收入、高速公路、人口所占比例
# 因变量：汽油消耗量
X = dataset[:, 0:6]
y = dataset[:, 6]

# 将数据分为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 特征缩放，通常没必要
# 因为数据单位，自变量数值范围差距巨大，不缩放也没问题
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练随机森林解决回归问题
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=500, random_state=42)
regressor.fit(X_train, y_train)
y_pred = np.round(regressor.predict(X_test))

# 评估回归性能
from sklearn import metrics
from sklearn.metrics import accuracy_score
print('平均绝对误差:', metrics.mean_absolute_error(y_test, y_pred))
print('均方误差:', metrics.mean_squared_error(y_test, y_pred))
print('均方根误差:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #越小越好
print(accuracy_score(y_test, y_pred))