import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from utils.model_architecture import SimpleMLP
from utils.client_utils import batch_data


def client_process(client_data: str, global_weights, comms_round):
    data_path = "./data/interim/"
    data = joblib.load(data_path + client_data)
    batched_data = batch_data(data)

    # create optimizer
    lr = 0.01 
    loss='categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(learning_rate=lr, 
                    decay=lr / comms_round, 
                    momentum=0.9
                ) 
    # build and compile model
    smlp_local = SimpleMLP()
    local_model = smlp_local.build(784, 10)
    local_model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=metrics)
    
    # set local model weight to the weight of the global model
    local_model.set_weights(global_weights)
    
    # fit local model with client's data
    local_model.fit(batched_data, epochs=1, verbose=0)
    
    # get model weights
    local_weights = local_model.get_weights()

    local_count = tf.data.experimental.cardinality(batched_data).numpy()*\
        (list(batched_data)[0][0].shape[0])

    # clear session to free memory
    K.clear_session()
    return local_weights, local_count
