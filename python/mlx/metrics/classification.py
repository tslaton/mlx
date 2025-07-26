# Copyright © 2023-2025 Apple Inc.

"""Classification metrics for evaluating model performance."""

from typing import Union, Optional
import mlx.core as mx


def accuracy(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    normalize: bool = True,
    sample_weight: Optional[mx.array] = None
) -> Union[float, mx.array]:
    """
    Compute accuracy classification score.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) labels.
    y_pred : mx.array
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : mx.array, optional
        Sample weights.

    Returns
    -------
    score : float or mx.array
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx.metrics import accuracy
    >>> y_true = mx.array([0, 1, 2, 3])
    >>> y_pred = mx.array([0, 2, 1, 3])
    >>> accuracy(y_true, y_pred)
    0.5
    >>> accuracy(y_true, y_pred, normalize=False)
    2
    """
    raise NotImplementedError("accuracy metric will be implemented in Task 5")


def precision(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    labels: Optional[mx.array] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[mx.array] = None,
    zero_division: Union[str, float] = "warn"
) -> Union[float, mx.array]:
    """
    Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated targets as returned by a classifier.
    labels : mx.array, optional
        The set of labels to include when ``average != 'binary'``.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary'} or None, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : mx.array, optional
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division.

    Returns
    -------
    precision : float or mx.array of shape (n_unique_labels,)
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.
    """
    raise NotImplementedError("precision metric will be implemented in Task 5")


def recall(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    labels: Optional[mx.array] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[mx.array] = None,
    zero_division: Union[str, float] = "warn"
) -> Union[float, mx.array]:
    """
    Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated targets as returned by a classifier.
    labels : mx.array, optional
        The set of labels to include when ``average != 'binary'``.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary'} or None, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : mx.array, optional
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division.

    Returns
    -------
    recall : float or mx.array of shape (n_unique_labels,)
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.
    """
    raise NotImplementedError("recall metric will be implemented in Task 5")


def f1_score(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    labels: Optional[mx.array] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = "binary",
    sample_weight: Optional[mx.array] = None,
    zero_division: Union[str, float] = "warn"
) -> Union[float, mx.array]:
    """
    Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated targets as returned by a classifier.
    labels : mx.array, optional
        The set of labels to include when ``average != 'binary'``.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary'} or None, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : mx.array, optional
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division.

    Returns
    -------
    f1_score : float or mx.array of shape (n_unique_labels,)
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.
    """
    raise NotImplementedError("f1_score metric will be implemented in Task 5")