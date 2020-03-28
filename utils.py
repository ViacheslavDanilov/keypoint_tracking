import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.metrics import f1_score

def get_random_dcm(dir):
    dcm = os.path.join(dir, random.choice(os.listdir(dir)))
    while os.path.isdir(dcm):
        dcm = os.path.join(dcm, random.choice(os.listdir(dcm)))
    return dcm

def macro_f1(y_true, y_pred, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    Args:
        y_true (int32 Tensor): labels array
        y_pred (float32 Tensor): probability matrix from forward propagation
        thresh: probability value above which a model predicts positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def f1_loss(y_true, y_pred):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y_true (int32 Tensor): targets array
        y_pred (float32 Tensor): probability matrix from forward propagation

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1                              # reduce (1 - f1) in order to increase f1
    macro_cost = tf.reduce_mean(cost)               # average on all labels
    return macro_cost

# def micro_f1_new(y_true, y_pred, thresh=0.5):
#     f1 = tfa.metrics.F1Score(num_classes=3, average='micro', name='Privet')
#     y_true = tf.cast(y_true, dtype=tf.int32)
#     y_pred = tf.cast(tf.greater(y_pred, thresh), tf.int32)
#     y_true = tf.one_hot(y_true, depth=2)
#     y_pred = tf.one_hot(y_pred, depth=2)
#     f1.update_state(y_true, y_pred)
#     # print('F1 Score is: ', f1.result().numpy())
#     return f1.result().numpy()
#
# def micro_f1_new_scikit(y_true, y_pred, thresh=0.5):
#     y_pred = tf.keras.backend.get_value(y_pred)
#     y_pred = (y_pred > thresh).astype('int')
#     y_true = tf.keras.backend.get_value(y_true)
#     # y_true = y_true.numpy()
#     y_true = y_true.astype('int')
#     f1 = f1_score(y_true, y_pred, average='micro')
#     return f1

def get_timing(sec):
    """Function that converts time period in seconds into %h:%m:%s expression.
    Args:
        sec (float): time period in seconds
    Returns:
        output (string): formatted time period
    """
    sec = int(sec)
    h = sec // 3600
    m = sec % 3600 // 60
    s = sec % 3600 % 60
    output = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
    return output

def learning_curves(history, fname):
    """Plot the learning curves of loss and macro f1 score
    for the training and validation datasets.

    Args:
        history: history callback of fitting a tensorflow model
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']

    epochs = len(loss)

    style.use("seaborn-whitegrid")
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), loss, label='Loss (Training)')
    plt.plot(range(1, epochs + 1), val_loss, label='Loss (Validation)')
    plt.ylim(0, 1)
    plt.ylabel('Loss')
    plt.title('Loss Dynamics')
    plt.legend(loc='upper right', frameon=True)

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), macro_f1, label='Macro F1-score (Training)')
    plt.plot(range(1, epochs + 1), val_macro_f1, label='Macro F1-score (Validation)')
    plt.ylim(0, 1)
    plt.ylabel('Macro F1-score')
    plt.title('Macro F1-score Dynamics')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(fname=fname, dpi=300, quality=100,
                bbox_inches='tight', pad_inches=0.05,
                facecolor='w', edgecolor='w', orientation='landscape')
    # plt.show()
    return loss, val_loss, macro_f1, val_macro_f1

def perfomance_grid(ds, target, label_names, model, save_dir, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn, tn,
    precision, recall and f-score metrics for each label.

    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int): number of thresholds to compute
        save_dir (str): path where the perfomance xlsx file is saved
    Returns:
        grid (Pandas dataframe): performance table
    """

    # Get model predictions
    y_pred_vals = model.predict(ds)
    y_pred_vals = y_pred_vals[0]
    # Define target matrix
    y_true_vals = target
    # Find label frequencies in the validation set
    label_freq = target.sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, tns, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], [], []
    for idx in label_index:
        for thresh in thresholds:
            ids.append(idx)
            labels.append(label_names[idx])
            freqs.append(round(label_freq[idx] / len(y_true_vals), 2))
            y_pred = y_pred_vals[:, idx]
            y_true = y_true_vals[:, idx]
            y_pred = y_pred > thresh
            tp = np.count_nonzero(y_pred * y_true)
            tn = np.count_nonzero((1 - y_pred)*(1 - y_true))
            fp = np.count_nonzero(y_pred * (1 - y_true))
            fn = np.count_nonzero((1 - y_pred) * y_true)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'ID': ids,
        'Label': labels,
        'Frequency': freqs,
        'Threshold': list(thresholds) * len(label_index),
        'TP': tps,
        'TN': tns,
        'FP': fps,
        'FN': fns,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s})

    grid = grid[['ID', 'Label', 'Frequency', 'Threshold', 'TP', 'TN', 'FN', 'FP', 'Precision', 'Recall', 'F1']]
    grid_name = os.path.join(save_dir, 'perfomance_grid.xlsx')
    grid.to_excel(grid_name, sheet_name='Perfomance', index=True, startrow=0, startcol=0)
    return grid

# ------------------------------------------------------- Handler ------------------------------------------------------
if __name__ == '__main__':
    # y_pred = tf.constant([[0.85, 0.15, 0.70], [0.65, 0.35, 0.27]])
    # y_true = tf.constant([[1.00, 0.00, 1.00], [0.00, 1.00, 0.00]])
    # y_pred = tf.constant([0.65, 0.35, 0.27])
    # y_true = tf.constant([0.00, 1.00, 0.00])
    # b = micro_f1_new(y_true, y_pred)
    # b_scikit = micro_f1_new_scikit(y_true, y_pred)
    # b_tf = micro_f1_wrong_axis(y_true, y_pred)
    print('Complete!')
