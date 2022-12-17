import argparse
import keras
import networkx as nx
import pandas as pd
import stellargraph as sg
import random as python_random
import tensorflow as tf

from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator


def main(training):
    # seeding
    python_random.seed(123)
    tf.random.set_seed(123)

    # load data first
    citations, word_attributes, labels = load_data()

    # get number of features
    num_features = len(labels.unique())

    # fit encoding of subjects (strings) to numerical target (one-hot encoding)
    encoding = preprocessing.LabelBinarizer()
    encoding.fit_transform(labels)

    # create generator based on citation network and word attributes
    generator = create_generator(citations, word_attributes)

    # train the model if necessary, otherwise try to load model from file
    if training:
        model, fold_accuracies = train_model(generator, labels, encoding, num_features)

        # Printing
        print("Average accuracy over all folds: {}".format(round(sum(fold_accuracies) / len(fold_accuracies), 4)))

        # save model
        model.save("save/model/")
    else:
        # try to load model
        try:
            model = keras.models.load_model("save/model/")
        except OSError:
            print("Could not load model. Please make sure that there is a correct model in the /save/model/ directory.")
            exit()

    predict_data(model, generator, labels, encoding)


def load_data():
    """
    Loads the cora dataset, renames the columns and extracts the features. 

    Returns
    -------
    citations : pandas.DataFrame
        The citation network of the cora.cites file.

    word_attributes : pandas.DataFrame
        The word attributes of the cora.content file.

    labels : pandas.Series
        The ground truth labels (paper subjects) of each paper from the cora.content file.
    """

    citations = pd.read_csv("dataset/cora.cites", sep="\t", header=None, names=["target", "source"])
    citations["label"] = "cites"

    # rename columns of word attributes
    word_attributes = ["w_{}".format(i) for i in range(1, 1434)]

    column_names = word_attributes + ['subject']
    papers = pd.read_csv("dataset/cora.content", sep="\t", header=None, names=column_names)
    word_attributes = papers[word_attributes]

    labels = papers["subject"]

    return citations, word_attributes, labels


def create_generator(citations, word_attributes):
    """
    Creates a FullBatchNodeGenerator based on a graph, which contains the citation network and word embeddings of the different papers.

    Parameters
    ----------
    citations : pandas.DataFrame
        The citation network of the cora.cites file.

    word_attributes : pandas.DataFrame
        Word embeddings of the papers.


    Returns
    -------
    generator : stellargraph.FullBatchNodeGenerator
        Generator which is used to generate the data batches for training.
    """

    # First, create a graph based on the citation network with networkx
    nGraph = nx.from_pandas_edgelist(citations, edge_attr='label')
    nx.set_node_attributes(nGraph, 'paper', 'label')

    # convert graph with labelled edges
    stellargraph = sg.StellarGraph.from_networkx(nGraph, node_features=word_attributes)

    # finally, create the generator based on the graph
    generator = FullBatchNodeGenerator(stellargraph, method="gcn")

    return generator


def create_model(generator, num_features) -> Model:
    """
    Creates a Graph Convolution Network which can be used for graph classification.

    Parameters
    ----------
    generator : stellargraph.FullBatchNodeGenerator
        Generator used for data generation based on the citation network and the paper word attributes.

    num_features : int
        Depicts the number of features (subjects).

    Returns
    -------
    model : keras.Model
        The graph convolution neural network as a keras model.
    """

    # initialize Graph Convolutional Network
    # modified model taken from https://stellargraph.readthedocs.io/en/latest/demos/graph-classification/gcn-supervised-graph-classification.html
    gc_model = GCN(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )

    input_tensor, output_tensor = gc_model.in_out_tensors()
    tmp = Dense(units=32, activation="relu")(output_tensor)
    tmp = Dense(units=16, activation="relu")(tmp)
    output = Dense(units=num_features, activation="softmax")(tmp)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["accuracy"])

    return model


def train_model(generator, labels, encoding, num_features, epochs: int = 100):
    """
    Parameters
    ----------
    generator : stellargraph.FullBatchNodeGenerator
        Generator used for data generation based on the citation network and the paper word attributes.

    labels : pandas.Series
        The ground truth labels (paper subjects) of each paper from the cora.content file.

    encoding: preprocessing.LabelBinarizer
        Encoder used in order to map subject string to numerical target (one-hot encoding).

    num_features : int
        Depicts the number of features (subjects).

    epochs : int
        Number of epochs the model is trained each fold.


    Returns
    -------
    model : keras.Model
        Trained model of the last fold.

    fold_accuracies : list
        List containing the accuracy scores obtained by the model during each fold.
    """

    fold_accuracies = []

    # initialize folds
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    for train, test in cv.split(labels, labels):
        # each fold depicts indices for training set and test set

        # get data of respective fold
        train_subjects = labels.iloc[train]
        test_subjects = labels.iloc[test]

        # get ground truth (subjects) for the data and transform to numerical target
        train_targets = encoding.fit_transform(train_subjects)
        test_targets = encoding.transform(test_subjects)

        # create a generator/sequence object for training
        training_generator = generator.flow(train_subjects.index, train_targets)
        test_generator = generator.flow(test_subjects.index, test_targets)

        # initialize the model
        model = create_model(generator, num_features)

        # train the model on the respective fold and save accuracy during the performance of the model on the test set
        accuracy = train_fold(model, training_generator, test_generator, epochs)
        fold_accuracies.append(accuracy)

    return model, fold_accuracies


def train_fold(model: keras.Model, training_generator, test_generator, epochs):
    """
    Trains a given keras model on a fold of a dataset which is given by two generators.

    Parameters
    ----------
    model : keras.Model
        The graph convolution neural network as a keras model.

    training_generator : stellargraph.FullBatchNodeGenerator
        Generator which is used to generate the training set according to the fold indices.

    test_generator : stellargraph.FullBatchNodeGenerator
        Generator which is used to generate the test set according to the fold indices.

    epochs : int
        Depicts the number of epochs for training.


    Returns
    -------
    fold_accuracy : float
        Accuracy of the model evaluated on the test set of the current fold.

    """
    model.fit(
        training_generator,
        epochs=epochs,
        verbose=0,
        validation_data=test_generator,
        shuffle=False
    )

    evaluation = model.evaluate(test_generator, verbose=0)
    fold_accuracy = evaluation[model.metrics_names.index("accuracy")]

    return fold_accuracy


def predict_data(model: keras.Model, generator, labels, encoding):
    """
    Predicts the whole data set and computes the subset accuracy.

    Parameters
    ----------
    model : keras.Model
        Depicts the neural network model.

    generator : stellargraph.FullBatchNodeGenerator
        Generator used for data generation based on the citation network and the paper word attributes.

    labels : pandas.Series
        The ground truth labels (paper subjects) of each paper from the cora.content file.

    epochs : int
        Number of epochs the model is trained each fold.
    """

    all_predictions = model.predict(generator.flow(generator.node_list))
    predictions = encoding.inverse_transform(all_predictions.squeeze())

    predicted_subjects = pd.DataFrame({"Paper_ID": labels.index, "Predicted": predictions})

    predicted_subjects.to_csv('predictions.tsv', sep="\t")

    score = accuracy_score(labels, predictions, normalize=True)
    print("Subset accuracy: {}".format(score))


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser("Script to train and predict subjects of paper.")

    parser.add_argument(
        "--train",
        default=argparse.SUPPRESS,
        help="Depicts whether a new model should be trained.",
    )

    training = False
    args = parser.parse_args()

    if "train" in args:
        training = True

    main(training)
