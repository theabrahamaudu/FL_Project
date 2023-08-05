"""
Model architecture module
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense


class SimpleMLP:
    """Simple Multi-Layer Perceptron model class
    """
    @staticmethod
    def build(shape: int, classes: int):
        """Takes the shape of the data, number of output classes
        and returns an MLP model to hadle such specifications.

        Args:
            shape (int): Shape if the input data. expected as 1D.
            classes (int): Number of output classes to be predicted.

        Returns:
            TensorFlow Model: MLP model to handle defined specifications
        """
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model