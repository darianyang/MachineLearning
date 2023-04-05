Introduction
In this homework, you will build a multitask neural network to predict the biological activity of small molecules for the proteins from the Janus Kinase family -- JAK1, JAK2, JAK3, and TYK2.

Janus kinases (Jaks) are critical signaling elements for a large subset of cytokines. As a consequence, they play pivotal roles in the pathophysiology of many diseases including neoplastic and autoimmune diseases.

Biological activity is expressed as a binary value -- a molecule can be "active" or "not active" for a protein of interest. The training data also contains missing endpoints since not all molecules were tested against all four proteins. However, test data does not contain missing endpoints.

Multitask Learning
Multitask Learning is an approach to inductive transfer that improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias. It does this by learning tasks in parallel while using a shared representation; what is learned for each task can help other tasks be learned better.

In this assignment, each task is a biological activity for one of the 4 JAK kinases. Since these proteins are evolutionary related and have similar binding domains, information about endpoints for all proteins can improve the model's performance due to the knowledge transfer.

Assignment
Your goal is to build a Character-based multitask Neural Network that learns a shared representation for an input molecule to predict its bioactivity profile. You will need to implement a multitask loss function, a weighted sum of 4 binary cross-entropy losses.

Introduction

In this homework, you will build a multitask neural network to predict the biological activity of small molecules for the proteins from the Janus Kinase family -- JAK1, JAK2, JAK3, and TYK2.

Janus kinases (Jaks) are critical signaling elements for a large subset of cytokines. As a consequence, they play pivotal roles in the pathophysiology of many diseases including neoplastic and autoimmune diseases.

Biological activity is expressed as a binary value -- a molecule can be "active" or "not active" for a protein of interest. The training data also contains missing endpoints since not all molecules were tested against all four proteins. However, test data does not contain missing endpoints.

Your goal is to build a Character-based multitask Neural Network that learns a shared representation for an input molecule to predict its bioactivity profile. You will need to implement a multitask loss function, a weighted sum of 4 binary cross-entropy losses.

 

Address

The HW3 is in the format of the Kaggle Competition. All data and instructions can be found here:

https://www.kaggle.com/t/1f2f7a084180485da72dd266963d3feeLinks to an external site. 

 

Submission: You will submit 3 objects: 

A pdf report with the following components: abstract (15 points),  background (15 points), your methods (30 points), the results and the analysis (30 points), and conclusion (10 points). The maximum length of your pdf report is 4 pages. This should be submitted to GradeScope HW3: report;
Submit your final code to GradeScope HW3: code;
Submit your predictions to Kaggle. Bonus points will be given for top submissions on the leaderboard. You can submit your predictions up to 10 times a day.

Evaluation
Your model’s performance will be evaluated using mean multi-column AUC metrics for 4 predicted biological activities. Please submit your predictions for the test as a .csv file. Each line in the submission file must contain a SMILES string followed by 4 columns with predicted classes (0 for "not active" and 1 for "active").

Dataset Description
Train data consists of 92921 molecules in the form of simplified molecular-input line-entry system (SMILES) strings. Each SMILES string is essentially a string of characters that encodes a molecular graph, including types of atoms and bonds in a molecule. The relation between SMILES strings and molecules is surjective, meaning that each SMILES string encodes a molecule, but one molecule can be encoded with multiple SMILES strings. This fact can be utilized to add data augmentation to the training pipeline. However, in the simplest scenario, you can treat SMILES just as a sequence of characters in a natural language for this assignment.

The bioactivity profile of each molecule is described by a binary vector of size 4 – 1 label per protein. Not all molecules were tested against all 4 proteins, so this data contains missing endpoints.

You will receive a .csv file train.csv with experimental data containing 5 columns – SMILES and 4 columns for bioactivity profile. Active molecules are marked as “1”, non-active are marked as “0” and missing endpoints are marked as "999".

You will also receive a .csv file test.csv with test SMILES strings.

Additionally, you will receive a random_submission.csv file showing a correct submission format.

A more detailed description of the files is below:

train.csv -- the training set of experimental data containing 5 columns:

StdSMILES -- SMILES string for molecules
P23458 -- labels for JAK1 bioactivity
O60674 -- labels for JAK2 bioactivity
P52333 -- labels for JAK3 bioactivity
P29597 -- labels for TYK2 bioactivity
test.csv -- the test set containing 1 column -- StdSMILES with test case molecules.
random_submission.csv - a sample submission file showing the correct format. This file contains the same 5 columns as train_jak.csv. Column StdSMILES must contain SMILES from test.csv, the rest 4 columns must contain your bioactivity profile predictions -- "1" for active and "0" for "not active".
