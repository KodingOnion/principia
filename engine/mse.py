def mse_loss(predictions, targets):
    """
    Calculates the Mean Squared Error between two Tensors.
    """
    # 1. Calculate the squared differences
    differences = predictions - targets
    squared_differences = differences ** 2
    
    # 2. Sum them all up (using your new method!)
    total_loss = squared_differences.sum()
    
    # 3. Divide by the total number of elements to get the mean
    # (We use targets.data.size to get the total number of items in the numpy array)
    n = targets.data.size
    mean_loss = total_loss * (1.0 / n)
    
    return mean_loss