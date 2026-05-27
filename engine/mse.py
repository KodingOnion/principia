def mse_loss(predictions, targets):
    """Compute mean squared error between ``predictions`` and ``targets``.

    Returns a scalar ``Tensor`` representing the mean of squared differences.
    """
    differences = predictions - targets
    squared_differences = differences ** 2
    total_loss = squared_differences.sum()
    n = targets.data.size
    mean_loss = total_loss * (1.0 / n)
    return mean_loss