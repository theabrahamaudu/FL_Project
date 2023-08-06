Federated Learning Project
==============================

Federated Learning project with TensorFlow

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── client.py
    │   │   └── server.py
    │   │
    │   └── utils          <- Scripts to create exploratory and results oriented visualizations
    │       ├── client_utils.py
    │       ├── model_architecture.py
    │       └── server_utils.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
# Getting Started
## Overview
This project contains a federated learning model approach to model building where the model is sent to the data and the data is ditributed across multiple clients as opposed to traditional model development where data is aggregated in a central storage for model training to take place.

The project was built using Tensorflow and is more conceptual in that the central server and lcient modules are separated in principle and model weights are transferred during the federated learning process, but the communication is direct rather than over a network as it would be in real-world depoloyment.

The primary purpose of the project is to demonsrate that models can be trained using private data on client machines without the central server or global model ever actually sseeing the data on the client devices. 

From a security perspective, this architechture can be taken a step further in real world applications by communicating the model weights between the client machines and the central server via a blockchain network to ensure the security and integrity of the model training process.

## Requirements
- Windows 10, Ubuntu, any other Python supported OS
- Python 3.10

## Experimenting
To start experimenting with this repository, follow the following steps:
- Clone this repository by running: ```git clone https://github.com/theabrahamaudu/FL_Project.git .```
- Create a virtual evironment (optional, recommended)
- Install the required modules: ```pip install -r requirements.txt```
- Download the dataset [here](https://drive.google.com/file/d/1MIAaSVbohQJszAXjIxkWiup_8zZ2QB25/view?usp=drive_link) and extract it in the data directory on the root folder
- Run the complete pipeline: ```python main.py```

Congratulations! You have now replicated the project setup.