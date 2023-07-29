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


def scale_weights(local_weights, scaling_factor):
    '''function for scaling a models weights'''
    weights_final = []
    steps = len(local_weights)
    for i in range(steps):
        weights_final.append(scaling_factor * local_weights[i])
    return weights_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss