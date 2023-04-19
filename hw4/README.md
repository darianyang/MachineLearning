HW4 Graph Neural Network

Due May 3 by 6pm Points 105 Available after Apr 17 at 8am

Description
A graph neural network (GNN) is a type of neural network that is specifically designed to work with graph data structures. Graphs are mathematical structures that consist of nodes (also known as vertices) and edges (also known as links) that connect them. GNNs can learn to extract features from graphs and make predictions based on them.

In molecular science, GNNs have become increasingly popular for tasks such as predicting molecular properties, discovering new molecules, and designing new drugs. Molecules can be represented as graphs, where atoms are represented as nodes and bonds between atoms are represented as edges. GNNs can be trained on large datasets of molecular graphs and their associated properties, allowing them to learn to predict properties of new molecules that they have not seen before.

In this homework, you are required to implement a graph-based neural network model to predict the target property of a graph(Regression).

There are two datasets in the data folder: train.pt, test.pt. You will train a GNN on the training dataset, then predict the graph property on the test dataset.

This HW needs to be implemented with Pytorch Geometric (PyG): https://pytorch-geometric.readthedocs.io/en/latest/
 
Evaluation
This is a regression task. The evaluation metric for this competition is **Root Mean Square Error (RMSE)**.


Address

The HW4 is in the format of the Kaggle Competition. All data and instructions can be found here:

https://www.kaggle.com/t/8ad6bea774d5453fa4873d8a318124c5Links to an external site.

 

Submission: You will submit 3 objects: 

A pdf report. The maximum length of your pdf report is 4 pages. This should be submitted to GradeScope HW4: report;
Submit your final code to GradeScope HW4: code;
Submit your predictions to Kaggle. Bonus points will be given for top submissions on the leaderboard. You can submit your predictions up to 10 times a day.


Dataset Description
Given the training data: **train.pt **and test.pt

You are allowed to build any type of CNN regression model using PyG (PyTorch Geometric)!
You are allowed to use any type of data processing (data augmentation).
Please explore training from scratch or finetuning/pertaining existing model.
Find the best performing model
Use your model to score test.pt
The performance will be measured using RMSE score. Overfitting is prevented by using public and private leaderboards (50%-50%).

train.pt - the training set. It's a list of graphs with correct labels. You can load the dataset withtorch.load('tarin.pt')
test.pt - the test set. It's a list of graphs with incorrect labels. You can load the dataset withtorch.load('tarin.pt'). You need to predict the correct labels with a trained graph neural network.

random_submission.csv - a sample submission file in the correct format
Columns
Idx - sample ID
labels - property value
