import numpy as np
import logging
from tqdm import tqdm

from itertools import product

logger = logging.getLogger(__name__)


def calc_score(refdiffs, diffs):
    """Calculate the significance score given the refdiffs and true diffs."""
    # the test statistics
    n = len(diffs)
    diffs = np.stack(diffs)
    if np.ndim(refdiffs) == 0:
        refdiffs = np.array([refdiffs])
    different = diffs > refdiffs
    same = np.isclose(diffs, refdiffs)

    ngecounts = np.sum(np.logical_or(different, same), 0)
    ngecounts = ngecounts.astype(np.float32)
    # If the labeling is not exact, we have to add 1 to
    # the counts.
    return (ngecounts + 1) / (n + 1)


def accuracy(gold, pred):
    """Get accuracy."""
    return np.sum(np.logical_and(gold, pred)).astype(np.float32) / np.sum(gold)


def precision_recall_fscore_micro(gold, pred):
    """Get micro-averaged precision, recall, f-score."""
    p = np.sum(np.logical_and(gold, pred)).astype(np.float32) / np.sum(gold)
    return np.stack([p, p, p])


def precision_recall_fscore_macro(gold, pred):
    """Get macro-averaged precision, recall, f-score."""
    temp = np.sum(np.logical_and(gold, pred)).astype(np.float32)
    p = np.nan_to_num(temp / pred.sum())
    r = np.nan_to_num(temp / gold.sum())
    f = np.nan_to_num(2*p*r / (p+r))

    return np.asarray([p, r, f])


def precision_recall_fscore_weighted(gold, pred):
    """Get weighted precision, recall, f-score."""
    gsum = gold.sum()
    gsum_div = gsum / np.sum(gold)

    temp = np.sum(np.logical_and(gold, pred)).astype(np.float32)
    p = np.nan_to_num(temp / pred.sum()) * gsum_div
    r = np.nan_to_num(temp / gsum) * gsum_div
    f = np.nan_to_num(2*p*r / (p+r))

    return np.asarray([p, r, f])


def teststatistic(gold,
                  system1,
                  system2,
                  scoring=precision_recall_fscore_micro,
                  absolute=True):
    """
    Compute the scores of 2 systems agains gold, and return the difference.
    Parameters
    ==========
    gold : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels). This represents
        the one-hot encoded labels of the gold standard.
    system1 : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels).
    system2 : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels).
    scoring : function, optional, default precision_recall_fscore_micro
        The function that calculates the score. This function should take
        as input two 2D matrices.
    absolute: boolean, optional, default True.
        Whether to take the absolute value of the difference between the
        two system scores.
    Returns
    =======
    diff : numpy array
        A 2D array, shaped (batch_size * num_scores), containing the scores
        of each system in the batch.
    """
    # Get the reference performance difference
    scores1 = scoring(gold, system1)
    scores2 = scoring(gold, system2)

    diff = scores1 - scores2
    if absolute:
        return np.abs(diff)

    return diff


def get_combinations(n):
    """Get 2**n items of length n."""
    return product((0, 1), repeat=n)


def _sublabelingloop(gold,
                     system1,
                     system2,
                     scoring,
                     shuffles,
                     absolute,
                     n,
                     return_distribution):
    """
    Inner loop of label-based ART.
    Parameters
    ==========
    gold : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels).
    system1 : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels).
    system2 : Numpy array
        A 2D numpy array, shaped (num_instances * num_labels).
    scoring : function
        The function which is used to score.
    common : Numpy array or list
        The items which are shared between both systems. These items are
        separated from system1 and system2 because these do not need to be
        shuffled. They are added to the shuffles.
    common_gold : Numpy array or list
        The items which are shared between both systems and gold.
    absolute : bool
        Whether to score with absolute values.
    n : The number of shuffles, so that we don't have to count.
    Returns
    =======
    proba : Numpy array
        A probability of failing to reject H0 for each of the scores.
    """
    if np.ndim(system1) != 2:
        raise ValueError("The labels of system 1 are not shaped like a 2D "
                         "array: {}".format(system1.shape))

    if np.ndim(system2) != 2:
        raise ValueError("The labels of system 2 are not shaped like a 2D "
                         "array: {}".format(system2.shape))

    refdiffs = teststatistic(gold,
                             system1,
                             system2,
                             scoring=scoring,
                             absolute=absolute)

    systems = np.array([system1, system2])
    column_indexer = np.arange(len(system1))

    diffs = []
    z = 0
    for shuf in tqdm(shuffles, total=n):

        shuffle1 = systems[shuf, column_indexer]
        if np.all(shuffle1 == system1):
            diffs.append(refdiffs)
            z += 1
            continue

        shuffle2 = systems[~shuf, column_indexer]

        diffs.append(teststatistic(gold,
                                   shuffle1,
                                   shuffle2,
                                   scoring=scoring,
                                   absolute=absolute))

    if return_distribution:
        return np.array(refdiffs), np.array(diffs)
    else:
        return calc_score(np.array(refdiffs,), np.array(diffs))


def exactlabelingsignificance(gold,
                              system1,
                              system2,
                              absolute,
                              scoring,
                              return_distribution):
    """Exact randomization testing."""
    shuffles = get_combinations(len(system1))
    n = 2**len(system1)

    refdiffs, diffs = _sublabelingloop(gold,
                                       system1,
                                       system2,
                                       scoring,
                                       shuffles,
                                       absolute=absolute,
                                       n=n,
                                       return_distribution=return_distribution)


def labelingsignificance(gold,
                         system1,
                         system2,
                         n,
                         absolute,
                         scoring,
                         return_distribution):
    """Approximate randomization testing."""
    shuffles = (np.random.randint(0, 2, len(gold)) for x in range(n))

    return _sublabelingloop(gold,
                            system1,
                            system2,
                            scoring,
                            shuffles,
                            absolute=absolute,
                            n=n,
                            return_distribution=return_distribution)