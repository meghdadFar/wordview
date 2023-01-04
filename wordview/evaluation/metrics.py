import pandas as pd
import numpy as np
import itertools


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from collections import namedtuple
from sklearn.metrics import precision_score, recall_score


def measure_accuracy(data_frame):
    data_frame['right_prediction'] = data_frame.apply(right_prediction, axis=1)
    num_correct_predictions = data_frame.right_prediction.sum()
    accuracy = float(num_correct_predictions)/data_frame.shape[0]
    return accuracy, num_correct_predictions


def right_prediction(row):
        if row['label'] == row['prediction']:
            return 1
        else:
            return 0


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_model(predictions, groundtruth, confustion_matrix_path=False):

    eval_df = pd.DataFrame(columns=['prediction', 'label'])
    eval_df['prediction'] = predictions
    eval_df['label'] = groundtruth

    accuracy, num_correct_predictions = measure_accuracy(eval_df)

    eval_df.sort_values(by=['label'], ascending=[True], inplace=True)
    
    if confustion_matrix_path:
        cnf_matrix = confusion_matrix(eval_df['label'], eval_df['prediction'])
        np.set_printoptions(precision=2)

        plt.figure(figsize=(30, 20))
        plt.rc('font', size=18)
        plot_confusion_matrix(cnf_matrix, classes=eval_df['label'].unique(), normalize=False,
                            title='Confusion matrix')
        plt.savefig(confustion_matrix_path, bbox_inches='tight')

    EvalResults = namedtuple('EvalResults', 'macro_p micro_p macro_rec micro_rec accuracy num_correct_predictions confusion_plot')
    return EvalResults(micro_p=precision_score(eval_df['label'], eval_df['prediction'], average='micro'),
                       macro_p=precision_score(eval_df['label'], eval_df['prediction'], average='macro'),
                       micro_rec=recall_score(eval_df['label'], eval_df['prediction'], average='micro'),
                       macro_rec=recall_score(eval_df['label'], eval_df['prediction'], average='macro'),
                       accuracy=accuracy,
                       num_correct_predictions=num_correct_predictions, 
                       confusion_plot=plt)
