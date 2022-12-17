# Task for ML Interview

Your task is to develop a machine learning approach to predict the subjects of scientific papers.

## Dataset
The Cora dataset consists of 2708 scientific publications classified into one of seven classes (`Case_Based`, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`). The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download Link: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

Related Papers:
- [Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2003/lu:icml03/)
- [Prithviraj Sen, et al. "Collective classification in network data." AI Magazine, 2008.](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2008/sen:aimag08/)

## Task
Your task is to develop a machine learning approach to predict the subjects of scientific papers:

1. Load the data
2. Split the dataset using 10-fold cross validation
3. Develop a machine learning approach to learn and predict the subjects of papers
4. Store your predictions in a file as tab-separated values (TSV) in the format `<paper_id> <class_label>` where *class_label* is a string.
5. Evaluate your approach in terms of *subset accuracy* (also known as exact match ratio) indicating the percentage of samples that have their label classified correctly.