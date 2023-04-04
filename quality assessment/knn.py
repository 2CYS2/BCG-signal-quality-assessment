import np
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    filepath = np.loadtxt('D:/bcg_data/assess_result/new/module_all.txt')   # 数据文件路径
    X = filepath[:, 0:7]
    y = filepath[:,7]
#拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#2、数据建模---模型训练/测试/应用
#引入KNN分类算法
knn=KNeighborsClassifier(n_neighbors=1)   #n_neighbors=1表示k=1
 #knn.fit(x,y)对训练数据进行拟合
knn = knn.fit(x_train,y_train)
#2.3 模型测试---knn.score(x_test,y_test)
score_test=knn.score(x_test,y_test)
y_pred_test = knn.predict(x_test)
score_train=knn.score(x_train,y_train)
y_pred_train = knn.predict(x_train)

i = '训练集'
print(f'{i}  准确率为:',accuracy_score(y_train,y_pred_train))
print(f'{i}  精确率为:',precision_score(y_train,y_pred_train, average='weighted'))
print(f'{i}  召回率为:',recall_score(y_train,y_pred_train, average='weighted'))
print(f'{i}  F1指数为:', f1_score(y_train,y_pred_train, average='weighted'))

j = '测试集'
print(f'{j}  准确率为:',accuracy_score(y_test,y_pred_test))
print(f'{j}  精确率为:',precision_score(y_test,y_pred_test, average='weighted'))
print(f'{j}  召回率为:',recall_score(y_test,y_pred_test, average='weighted'))
print(f'{j}  F1指数为:', f1_score(y_test,y_pred_test, average='weighted'))