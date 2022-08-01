from sklearn import metrics
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns


def show_confusion_matrix(y_true, y_pred, labels, path):
    matrix = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 7))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(path + "confusion_matrix.png")
    plt.show()


def get_classification_report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, target_names=labels)


# ================ USAGE ================ #
#
# y_true = [0,1,2,3,4,5,6]
# y_pred = [6,5,4,3,2,1,0]
#
# label_mapping = {0: "Surprise",
#                      1: "Fear",
#                      2: "Disgust",
#                      3: "Happiness",
#                      4: "Sadness",
#                      5: "Anger",
#                      6: "Neutral"}
#
# labels_list = []
# for i in range(len(label_mapping)):
#     labels_list.append(label_mapping[i])
#
# show_confusion_matrix(y_true, y_pred, labels_list)
# print(get_classification_report(y_true, y_pred, labels_list))

