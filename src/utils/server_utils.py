"""
Untility functions for server module
"""

import os
import tensorflow as tf
from sklearn.metrics import accuracy_score

def scan_directory(directory: str, contains: str) -> list:
    """Check specified directory and return list of files with
    specified extension

    Args:
        directory (str): path string to directory e.g. "./the/directory"
        extension (str): extension type to be searched for e.g. ".csv"

    Returns:
        list: strings of file names with specified extension
    """    
    files: list = []
    for filename in os.listdir(directory):
        if contains in filename:
            files.append(filename)
    return files


def scale_weights(local_weights, scaling_factor) -> list:
    """Takes list of local model weights, scaling factor and scales
    the weights.

    Args:
        local_weights (list): local model weights
        scaling_factor (float): multiplier to scale model weights

    Returns:
        list: Scaled model weights
    """
    weights_final = []
    steps = len(local_weights)
    for i in range(steps):
        weights_final.append(scaling_factor * local_weights[i])
    return weights_final


def sum_scaled_weights(scaled_weight_list: list) -> list:
    """Takes a list of lists containing scaled weights 
    and returns a new list of scaled weights, where each
    weight is the average of the corresponding weights from different lists.

    Args:
        scaled_weight_list (list): A list of lists, each containing scaled weights.

    Returns:
        list: A list of averages, with each element representing the average of the 
        corresponding scaled weights.
    """
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round) -> tuple:
    """
    Evaluates the performance of a trained model on a test dataset.

    Args:
        X_test (numpy array): Test input data.
        Y_test (array): Ground truth labels for the test data.
        model (tf.keras.Model): The trained model to evaluate.
        comm_round (int): The communication round or iteration number.

    Returns:
        tuple: A tuple containing the accuracy and loss of the model on the test dataset.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss