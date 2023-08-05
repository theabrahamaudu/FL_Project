## Dataset
The MNIST dataset was used for training and evaluation of the federated learning model

## Preprocessing 
The images were loaded in greysacle and converted into flattened numpy arrays. Based on the structure of the MNIST dataset, the subdirectory names were used to generate the image labels. The images and corresponding labels were then added sequentially to sparate but identically indexed arrays.
To improve the classification process, the labels were converted into binary vectors using a label binarizer. The dataset was then randomized and slpit into train and test sets.

For the purpose of federated learning, the test set was then subdivided into four different sets otherwise termed client data for later use in simulating dsitributed training.

## Model Training, Aggregation and Evaluation
### Client side
The client side of the federated learning process closely resembles traditional model training process.
The major differences being that the intial local model weights are initialized with the global model weights and the model optimizer takes into account the number of global epochs being run for that training session in setting up the decay parameter to prevent overfitting. This is important because the local training process only runs for one epoch at a time, so it is important to regularize the model training across the global training epochs.
At the end of local training, the updated local weights and the total number of data points used for the round of training are sent back to the server.


### Server side
On the server side, the model architecture is initialized to as to define the model weights to be trained and the test data is also loaded, after which the federated learning process is initiated.
In the federated learning step, the client process for each client is initiated by sending the initialized global weights and related metadata to the client and the resulting client model weights and epoch data point count are retrived and stored. The model weights received from client machines are then scaled by the data point count for each client and aggregated by taking the reduced sum of each weight accross the different clients.
The global model is then updated with this new set of scaled average weights, after which the updated global model is evaluated using the test dataset. The new set of global weights is sent as the new set of seed weights to the client machines in the next federated training epoch. 
This process is repeated in a loop for a specified number of global training epochs.

### Further blurb (take what you need)
In this way, user data does not leave the client, rather, model weights are sent between the client and server. 
In practical use case, the model weight would be transmited via a standard HTTP or HTTPS connection. however, to ensure the security and integrity of the model metrics being transmitted, the metrics could be sent as trnasaction payloads over a crptographic blockchain network where the client pushes the metrics to the network and the public key or some other form of ID is then used as a reference to access the model metrics on the server side.


### Model Metrics
    132/132 [==============================] - 0s 2ms/step
    comm_round: 0 | global_acc: 90.857% | global_loss: 1.6153173446655273
    132/132 [==============================] - 0s 2ms/step
    comm_round: 1 | global_acc: 92.929% | global_loss: 1.5779491662979126
    132/132 [==============================] - 0s 2ms/step
    comm_round: 2 | global_acc: 94.024% | global_loss: 1.5585637092590332
    132/132 [==============================] - 0s 2ms/step
    comm_round: 3 | global_acc: 94.643% | global_loss: 1.5451048612594604
    132/132 [==============================] - 0s 1ms/step
    comm_round: 4 | global_acc: 95.476% | global_loss: 1.534764289855957
