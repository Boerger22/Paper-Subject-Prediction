# Paper-Subject-Prediction

The Cora data set consists of two files. The first file `cora.content` consists of 2708 scientific publications in the form `<paper_id> <word_attributes>+ <class_label>`. Thus, for each paper, there is a unique PaperID, and also a dictionary of 1433 words that describes whether a certain word occurs in the paper or not. Finally, the file contains the respective subjects of the papers. 

The second file `cora.cites` represents a citation network as a graph in the form `<ID of cited paper> <ID of citing paper>`, where the first paper is cited by the second paper. In total, the network consists of 5429 connections.


# Overview
To develop a machine learning approach, which learns from this data and can predict the subject of a paper, an approach based on a network is reasonable. In doing so, one could model a network or graph where each paper represents a node and a connection of two nodes represents a citation. This is because it is very likely that a paper with a particular subject will also cite and be cited by other papers with the same subject. In addition, each node can be given further information in the form of the occurrences of the words from the Dictionary. This gives a graph that represents the citation network and includes information about the content of the papers in the form of word embeddings.

To build such a network, we can first create pandas dataframes and then create networks from these dataframes using the library [networkx](https://networkx.org/). In order to model the information of the nodes as well and to apply already existing machine learning algorithms to this network, we convert the networkx graph to a StellarGraph using the library [stellargraph](https://stellargraph.readthedocs.io/en/stable/). This library contains many graph machine learning algorithms, such as the Graph Convolutional Network (GCN), which are used to classify nodes within a network/graph. This library also provides a connection to the machine learning framework [TensorFlow](https://www.tensorflow.org/).

Thus, we can use the Cora dataset to model a network and then use a neural network to learn the classifications of the nodes based on the respective word embeddings and the connections (citations) to other papers to eventually be able to classify the subject of papers.

# Approach in more detail

We first load the two files of the Cora dataset and create a network based on this information. Then, we map the subject strings to numerical values to be able to learn and predict values instead of dealing with strings. For this purpose, we use the one-hot encoding of the different subjects. Afterward, we define a GCN model and train it with the network. Thereby, we use 10-fold cross-validation to estimate the performance of the network. For each fold, we train the network for 100 epochs. Once the model is trained, we compute the average accuracy across all 10 folds and save the trained model in `save\model\`. In the end, we use the trained model to predict the whole network, which we save as a tab-separated values file in `save\` with format: `<paper_id> <predicted_class_label>`. We also compute the subset accuracy of the trained model.

After training our model for 100 epochs, we achieve a subset accuracy of about 0.9287. 

# Requirements

- python (>=v3.8, <v3.9)
- keras
- tensorflow
- networkx
- pandas
- stellargraph
- scikit-learn

See `requirements.txt` for used versions

# How to run

If you want to use the pre-trained model you can just execute the main file
```
python main.py
```

<br>

However, if you want to train a new model, you need to use the `-train` argument
```
python main.py -train
```
This trains a new GCN Model for 100 epochs and saves the model to `save\model\`.

<br>

When both commands are executed, either a new model is trained or the trained model in `save\model\` is used to predict the subjects of the paper of the Cora dataset. Furthermore, the script calculates the subset accuracy based on the model and stores the predictions are saved in `save\` as a .tsv file in the format: `<paper_id> <class_label>`