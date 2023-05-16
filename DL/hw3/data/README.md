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
