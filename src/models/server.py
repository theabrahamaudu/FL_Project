"""
Server module to initiate federated learning on client devices.
"""

import tensorflow as tf
import random
import joblib
from src.utils.model_architecture import SimpleMLP
from src.models.client import client_process
from src.utils.server_utils import (scan_directory,
                                scale_weights,
                                sum_scaled_weights,
                                test_model)


def server_process(comms_round: int):
    """Initialize global model, load test set, run federated learning
    loop across client devices for `comms_round` epochs.

    Args:
        comms_round (int): Number of global training rounds
    """

    #initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(784, 10)

    # Load and batch the test set
    data_path = "./data/interim"
    X_test = joblib.load(data_path+"/X_test.joblib")
    y_test = joblib.load(data_path+"/y_test.joblib")
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

    #commence global training loop
    for comm_round in range(comms_round):
                
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        
        #randomize client data
        client_data_path = "./data/interim"
        client_names= scan_directory(client_data_path, "client")
        random.shuffle(client_names)

        # initialize global data points count, client feedback dict
        global_count = 0
        client_feedback = {}

        # Initiate federated training
        for client in client_names:
            local_weights, local_count = client_process(
                client_data=client,
                global_weights=global_weights,
                comms_round=comms_round
            )
            global_count += local_count
            client_feedback[client]={"weights": local_weights, "count": local_count}
        
        # Initialize list to collect local model weights after scalling
        scaled_local_weight_list = list()

        # Get scaled weights for each client model
        for client in client_names:
            scaling_factor = client_feedback[client]["count"]/global_count
            scaled_weights = scale_weights(
                local_weights=client_feedback[client]["weights"],
                scaling_factor=scaling_factor
            )
            scaled_local_weight_list.append(scaled_weights)

        # Get average of weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # Update global model weights
        global_model.set_weights(average_weights)

        # Test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

if __name__ == "__main__":
    server_process(comms_round=5) # Run global training for 5 epochs