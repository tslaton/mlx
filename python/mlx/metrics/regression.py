# Copyright © 2023-2025 Apple Inc.

"""Regression metrics for evaluating model performance."""

from typing import Optional, Literal
import mlx.core as mx


def mean_squared_error(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    sample_weight: Optional[mx.array] = None,
    multioutput: Literal["raw_values", "uniform_average"] = "uniform_average",
    squared: bool = True
) -> mx.array:
    """
    Mean squared error regression loss.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated target values.
    sample_weight : mx.array, optional
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
        'raw_values' : Returns a full set of errors in case of multioutput input.
        'uniform_average' : Errors of all outputs are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or mx.array
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx.metrics import mean_squared_error
    >>> y_true = mx.array([3, -0.5, 2, 7])
    >>> y_pred = mx.array([2.5, 0.0, 2, 8])
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.6123724...
    """
    raise NotImplementedError("mean_squared_error metric will be implemented in Task 13")


def mean_absolute_error(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    sample_weight: Optional[mx.array] = None,
    multioutput: Literal["raw_values", "uniform_average"] = "uniform_average"
) -> mx.array:
    """
    Mean absolute error regression loss.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated target values.
    sample_weight : mx.array, optional
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
        'raw_values' : Returns a full set of errors in case of multioutput input.
        'uniform_average' : Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or mx.array
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx.metrics import mean_absolute_error
    >>> y_true = mx.array([3, -0.5, 2, 7])
    >>> y_pred = mx.array([2.5, 0.0, 2, 8])
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    """
    raise NotImplementedError("mean_absolute_error metric will be implemented in Task 13")


def r2_score(
    y_true: mx.array,
    y_pred: mx.array,
    *,
    sample_weight: Optional[mx.array] = None,
    multioutput: Literal["raw_values", "uniform_average", "variance_weighted"] = "uniform_average"
) -> mx.array:
    """
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Parameters
    ----------
    y_true : mx.array
        Ground truth (correct) target values.
    y_pred : mx.array
        Estimated target values.
    sample_weight : mx.array, optional
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, default='uniform_average'
        Defines aggregating of multiple output scores.
        'raw_values' : Returns a full set of scores in case of multioutput input.
        'uniform_average' : Scores of all outputs are averaged with uniform weight.
        'variance_weighted' : Scores of all outputs are averaged, weighted by the
            variances of each individual output.

    Returns
    -------
    z : float or mx.array
        The R^2 score or mx.array of scores if 'multioutput' is 'raw_values'.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx.metrics import r2_score
    >>> y_true = mx.array([3, -0.5, 2, 7])
    >>> y_pred = mx.array([2.5, 0.0, 2, 8])
    >>> r2_score(y_true, y_pred)
    0.948...
    """
    raise NotImplementedError("r2_score metric will be implemented in Task 13")