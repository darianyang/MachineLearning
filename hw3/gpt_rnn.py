"""
A further optimized ChatGPT output for a RNN.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import roc_auc_score
from SmileEnumerator import SmilesEnumerator 

class BioactivityDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['StdSMILES']
        bioactivity = torch.Tensor(self.data.iloc[idx][['P23458', 'O60674', 'P52333', 'P29597']].values)
        return smiles, bioactivity

class CharacterTokenizer:
    def __init__(self, padding_idx=0, character_vocab='!\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'):
        self.padding_idx = padding_idx
        self.character_vocab = character_vocab

        self.char2idx = {}
        self.idx2char = {}

        for i, char in enumerate(self.character_vocab):
            self.char2idx[char] = i + 1 # leave 0 for padding index
            self.idx2char[i + 1] = char # leave 0 for padding index

    def __len__(self):
        return len(self.char2idx)

    def __call__(self, sequence):
        return [self.char2idx[char] for char in sequence]

    def decode(self, sequence):
        return ''.join([self.idx2char[idx] for idx in sequence])

class MultiTaskCharacterRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MultiTaskCharacterRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        batch_size, seq_len, hidden_size = outputs.shape
        last_hidden = hidden[-1]
        last_hidden = last_hidden.view(batch_size, hidden_size)
        outputs = self.fc(last_hidden)
        return outputs

def collate_fn(batch):
    smiles_list, bioactivity_list = zip(*batch)

    tokenizer = CharacterTokenizer()
    smiles_lengths = torch.LongTensor([len(smiles) for smiles in smiles_list])
    smiles_padded = pad_sequence([torch.LongTensor(tokenizer(smiles)) for smiles in smiles_list], batch_first=True, padding_value=tokenizer.padding_idx)

    return smiles_padded, smiles_lengths, torch.stack(bioactivity_list)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (smiles, lengths, bioactivity) in enumerate(train_loader):
        smiles, bioactivity = smiles.to(device), bioactivity.to(device)
        optimizer.zero_grad()
        output = model(smiles, lengths)
        loss = criterion(output, bioactivity)
        loss.backward()
        optimizer.step()
        total_loss += loss.item


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from model import RNNMultitaskClassifier
from data_processing import process_data, MultitaskDataset, collate_fn

def main():
    # Set the random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the training and test data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    # Process the training and test data
    train_data, test_data, char_to_idx = process_data(train_df, test_df)
    
    # Define hyperparameters
    batch_size = 256
    hidden_size = 256
    num_layers = 3
    num_epochs = 20
    
    # Initialize the model and optimizer
    model = RNNMultitaskClassifier(len(char_to_idx), hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloaders for the training and test data
    train_loader = DataLoader(MultitaskDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(MultitaskDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            losses = [nn.BCEWithLogitsLoss()(outputs[j], targets[j]) for j in range(len(targets))]
            loss = sum(losses)
            
            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")
        
        # Print epoch loss
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Evaluate on the test set
        model.eval()
        auc_scores = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute AUC scores
            auc_scores += [roc_auc_score(targets[j].cpu().numpy(), torch.sigmoid(outputs[j]).cpu().numpy()) for j in range(len(targets))]
        
        # Print mean AUC scores
        print(f"Epoch {epoch + 1}, Mean AUC: {np.mean(auc_scores):.4f}")
    
    # Generate predictions for the test set
    model.eval()

