import numpy as np

EPS = 1e-20


def check_shape(func):
    def wrapper(*args, **kwargs):
        assert isinstance(args[0], np.ndarray)
        assert isinstance(args[1], np.ndarray)
        assert args[0].shape == args[1].shape
        return func(*args, **kwargs)
    return wrapper


@check_shape
def mae(out: np.ndarray, gt: np.ndarray):
    '''
        Mean absolute error between 2 images
    '''
    return np.mean(np.abs(out - gt))


@check_shape
def precision_recall(y_pred: np.ndarray, y: np.ndarray):
    """
        Precision and recall for two binary images
    """
    tp = (y_pred * y).sum()
    precision, recall = tp / (y_pred.sum() + EPS), tp / (y.sum() + EPS)
    return precision, recall


@check_shape
def f_beta_measure(pred: np.ndarray, gt: np.ndarray, beta=1):
    """
        F-beta score for binary images.
        In case of beta=1, F-beta converts to F-1 score
    """
    p, r = precision_recall(pred, gt)
    f_score = (1 + beta) * (p * r) / (beta * p + r + EPS)

    return f_score


@check_shape
def f_beta_score_with_threshold(pred: np.ndarray, gt: np.ndarray, threshold_num=255, beta=0.3):
    """
        Calculate F-beta score for several thresholds from 0 to 1.
        Threshold is used to convert prediction to binary image
    """
    result = dict()
    thresholds = np.linspace(0, 1, threshold_num)
    for t in thresholds:
        y_pred = (pred >= t).astype(np.float)
        f_score = f_beta_measure(y_pred, gt, beta)
        result[t] = f_score
    return result


def s_measure():
    """
        Calculates Structure-measure: https://arxiv.org/abs/1708.00786

    """
    pass


def e_measure():
    pass
