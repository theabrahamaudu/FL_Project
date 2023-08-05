"""
Overall FL Pipeline
"""

from src.data.make_dataset import (load,
                                   create_clients,
                                   save_test_set,
                                   paths,
                                   LabelBinarizer,
                                   train_test_split)
from src.models.server import server_process


## ----------------- Preprocess and Split Data -------------------- ##
# path to data folder
img_path = 'data/raw/trainingSet/trainingSet'

# get the path list using the path object
image_paths = list(paths.list_images(img_path))

# Load data
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

## --------------- Train and Evaluate FL Model ------------------- ##
server_process(comms_round=5) # Run global training and eval for 5 epochs