#随机森林分类
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    dataset = np.loadtxt(r'D:/bcg_data/assess_result/new/module_all.txt')
    X = dataset[:, 0:7]
    y = dataset[:,7]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    regressor = RandomForestRegressor(n_estimators=500, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred_train = np.round(regressor.predict(X_train))
    y_pred_test = np.round(regressor.predict(X_test))
    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)
    print ("训练集准确率:",accuracy_train)
    print ("测试集准确率:",accuracy_test)
    print(y_pred_test)


    feature_names = ['PSQI','iSQI','SNR','KSQI','SSQI','SampleEn','LZC']
    perm_importance = permutation_importance(regressor, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(np.array(feature_names)[sorted_idx], np.array(perm_importance.importances_mean)[sorted_idx])
    plt.xlabel("Permutation Importance")


    class_names = ['1','2', '3']
    print("训练集的性能评估：\n", classification_report(y_train, y_pred_train, target_names=class_names))
    print("测试集的性能评估：\n", classification_report(y_test, y_pred_test, target_names=class_names))
    a = cmpic(y_train, y_pred_train)
    b = cmpic(y_test, y_pred_test)
    pyplot.show()