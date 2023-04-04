import np
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def importance(name,j):
    if(j == 1):
        feature_names = ['PSQI', 'ISQI', 'SNR', 'KSQI', 'SSQI', 'BSQI', 'SampleEn','LZC']
    if (j == 2):
        feature_names = ['PSQI', 'ISQI', 'SNR', 'KSQI', 'SSQI', 'SampleEn','LZC']
    perm_importance = permutation_importance(name, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(np.array(feature_names)[sorted_idx], np.array(perm_importance.importances_mean)[sorted_idx])

# load data
dataset = np.loadtxt(r'D:/bcg_data/assess_result/new/double_all.txt')
X = dataset[:,0:8]
Y = dataset[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
predictions = [round(value) for value in y_pred_test]
score_train = accuracy_score(y_train, y_pred_train)
score_test = accuracy_score(y_test, y_pred_test)
print("训练集准确率: %.2f%%" % (score_train * 100.0))
print("测试集准确率: %.2f%%" % (score_test * 100.0))

# plot_importance(model)
importance(model,1)
# thresholds = sort(model.feature_importances_)
# for thresh in thresholds:
# 	# select features using threshold
# 	selection = SelectFromModel(model, threshold=thresh, prefit=True)
# 	select_X_train = selection.transform(X_train)
# 	# train model
# 	selection_model = XGBClassifier()
# 	selection_model.fit(select_X_train, y_train)
# 	# eval model
# 	select_X_test = selection.transform(X_test)
# 	y_pred = selection_model.predict(select_X_test)
# 	predictions = [round(value) for value in y_pred]
# 	accuracy = accuracy_score(y_test, predictions)
# 	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


def cmpic(y_true, y_pred):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
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
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,8), dpi=120)
    #set the fontsize of label.
    #for label in plt.gca().xaxis.get_ticklabels():
    #    label.set_fontsize(8)
    #text portion
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    #offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

# 计算precision, recall, F1-score, support
class_names = ['1', '2', '3']
print("训练集的性能评估：\n", classification_report(y_train, y_pred_train, target_names=class_names))
print("测试集的性能评估：\n", classification_report(y_test, y_pred_test, target_names=class_names))

a = cmpic(y_test, y_pred_test)
pyplot.show()