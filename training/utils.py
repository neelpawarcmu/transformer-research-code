def get_learning_rate(step_num, d_model, warmup):
    """
    Compute the learning rate from the equation (3) in section 5.3
    of the paper.
    """
    learning_rate = d_model**-0.5 * min(step_num**-0.5, step_num*warmup**-1.5)
    return learning_rate