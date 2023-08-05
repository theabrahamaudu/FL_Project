"""
Handles data loading and transformations
"""

import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import joblib


def load(paths: list, verbose: int = 10000) -> tuple:
    """Loads each image, converts it to a numpy array and adds the array to list.
       Adds source folder name as label on same index in separate labels list.

       expects images for each class in seperate dir, 
       e.g all digits in 0 class in the directory named 0

    Args:
        paths (list): list of numbered subdirs
        verbose (int, optional): Option to display load progress at given file threshold. Defaults to 10000.

    Returns:
        tuple: List of image arrays, List of image labels
    """
    
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels

def create_clients(image_list: list, label_list: list, num_clients: int=4, initial: str='clients') -> dict:
    """Takes list of numpy arrays and labels, randomize and splits data into number of parts specified 
    as `num_clients`, names each part (list of array-label tuples) with prefix specified as `initial` followed by the
    part number e.g. `clients_2`. Saves each client data to file.

    Args:
        image_list (list): List of image numpy arrays
        label_list (list): List of image labels
        num_clients (int, optional): Number of client datasets to create. Defaults to 4.
        initial (str, optional): Prefix string for client names. Defaults to 'clients'.

    Returns:
        dict: client names as keys and file path as values.
    """

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    _ = [joblib.dump(shards[i], f"./data/interim/{client_names[i]}.joblib") for i in range(len(client_names))]

    return {client_names[i] : f"./data/interim/{client_names[i]}.joblib" for i in range(len(client_names))}

def save_test_set(X_test, y_test) -> dict:
    """Save test arrays and labels to file

    Args:
        X_test (list): Image numpy arrays
        y_test (list): Lablels

    Returns:
        dict: Saved file paths
    """
    _ = joblib.dump(X_test, "./data/interim/X_test.joblib")
    _ = joblib.dump(y_test, "./data/interim/y_test.joblib")

    return {"X_test": "./data/interim/X_test.joblib",
            "y_test": "./data/interim/y_test.joblib"}

if __name__=="__main__":
    # path to data folder
    img_path = 'data/raw/trainingSet/trainingSet'

    # get the path list using the path object
    image_paths = list(paths.list_images(img_path))

    # apply our function
    image_list, label_list = load(image_paths, verbose=10000)

    # binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                        label_list, 
                                                        test_size=0.1, 
                                                        random_state=42)

    # create clients
    clients = create_clients(X_train, y_train, num_clients=4, initial='client')

    # save test set
    test_set = save_test_set(X_test=X_test, y_test=y_test)