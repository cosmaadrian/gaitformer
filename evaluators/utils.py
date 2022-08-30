import numpy as np

def log_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def hamming_loss(y_true, y_pred):
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    return hl_num/hl_den

def emr(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count/n

def example_based_accuracy(y_true, y_pred):
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator
    avg_accuracy = np.mean(instance_accuracy)
    return avg_accuracy

def example_based_precision(y_true, y_pred):
    precision_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)
    precision_den = np.sum(y_pred, axis = 1)
    avg_precision = np.mean(precision_num/precision_den)
    return avg_precision

def label_based_macro_accuracy(y_true, y_pred):
    l_acc_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)
    l_acc_den = np.sum(np.logical_or(y_true, y_pred), axis = 0)
    return np.mean(l_acc_num/l_acc_den)

def label_based_macro_precision(y_true, y_pred):
    l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)
    l_prec_den = np.sum(y_pred, axis = 0)
    l_prec_per_class = l_prec_num/l_prec_den
    l_prec = np.mean(l_prec_per_class)
    return l_prec

def label_based_macro_recall(y_true, y_pred):
    l_recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)
    l_recall_den = np.sum(y_true, axis = 0)
    l_recall_per_class = l_recall_num/l_recall_den
    l_recall = np.mean(l_recall_per_class)
    return l_recall

def label_based_micro_accuracy(y_true, y_pred):
    l_acc_num = np.sum(np.logical_and(y_true, y_pred))
    l_acc_den = np.sum(np.logical_or(y_true, y_pred))
    return l_acc_num/l_acc_den

def label_based_micro_precision(y_true, y_pred):
    l_prec_num = np.sum(np.logical_and(y_true, y_pred))
    l_prec_den = np.sum(y_pred)
    return l_prec_num/l_prec_den

def label_based_micro_recall(y_true, y_pred):
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    l_recall_den = np.sum(y_true)
    return l_recall_num/l_recall_den


def alpha_evaluation_score(y_true, y_pred):
    alpha = 1
    beta = 0.25
    gamma = 1
    tp = np.sum(np.logical_and(y_true, y_pred))
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    alpha_score = (1 - ((beta * fn + gamma * fp ) / (tp +fn + fp + 0.00001)))**alpha

    return alpha_score
