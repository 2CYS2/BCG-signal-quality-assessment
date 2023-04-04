import numpy as np
from matplotlib import pyplot
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from numpy import sort
# 导入数据，路径中要么用\\或/或者在路径前加r
from xgboost import XGBClassifier, plot_importance

dataset = np.loadtxt(r'D:/assess_result/new/double_all.txt')

# 准备训练数据
X = dataset[:, 0:8]
y = dataset[:, 8]

# 将数据分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 因为数据单位，自变量数值范围差距巨大，不缩放也没问题
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练随机森林解决回归问题

regressor = RandomForestRegressor(n_estimators=500, random_state=42)
regressor.fit(X_train, y_train)
y_pred = np.round(regressor.predict(X_test))

pyplot.bar(range(len(regressor.feature_importances_)), regressor.feature_importances_)
thresholds = sort(regressor .feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(regressor, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

class_names = ['1', '2', '3']
print(classification_report(y_test, y_pred, target_names=class_names))
print("每个类别的精确率和召回率：\n", classification_report(y_test, y_pred, target_names=class_names))

# 评估回归性能
print('平均绝对误差:', metrics.mean_absolute_error(y_test, y_pred))
print('均方误差:', metrics.mean_squared_error(y_test, y_pred))
print('均方根误差:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #越小越好
print(accuracy_score(y_test, y_pred))

pyplot.show()