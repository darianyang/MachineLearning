import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
import warnings
import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from rdkit import Chem
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import time
import math
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss #, _assert_no_grad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol*(max_length - cur_len)
    return seqs, lengths

def sanitize_smiles(smiles, canonicalize=True):
    new_smiles = []
    idx = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            if canonicalize:
                new_smiles.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=False))
                )
                idx.append(i)
            else:
                new_smiles.append(sm)
                idx.append(i)
        except: 
            warnings.warn('Unsanitized SMILES string: ' + sm)
            new_smiles.append('')
    return new_smiles, idx

def seq2tensor(seqs, tokens, flip=False):
    tensor = np.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + seqs[i][j]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens

def get_tokens(smiles, tokens=None):
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = np.sort(tokens)[::-1]
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


    
class SmilesDataset(Dataset):
    def __init__(self, filename, smiles_col, target_cols=None, tokens=None,
                 pad=True, tokenize=True, augment=False):
        super(SmilesDataset, self).__init__()
        self.tokenize = tokenize
        data = pd.read_csv(filename)
        smiles = data[smiles_col].values
        clean_smiles, clean_idx = sanitize_smiles(smiles)
        if target_cols is not None:
            target = data[target_cols].values
            self.target = target[clean_idx]
        else:
            self.target = None
        #if augment:
        #    clean_smiles, self.target = augment_smiles(clean_smiles,
        #                                               self.target)
        if pad:
            clean_smiles, self.length = pad_sequences(clean_smiles)
        tokens, self.token2idx, self.num_tokens = get_tokens(clean_smiles,
                                                             tokens)
        if tokenize:
            clean_smiles, self.tokens = seq2tensor(clean_smiles, tokens)
        self.data = clean_smiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}
        sample['tokenized_smiles'] = self.data[index]
        sample['length'] = self.length[index]
        if self.target is not None:
            sample['labels'] = self.target[index]
        return sample


    
class RNN_network(nn.Module):
    def __init__(self, n_layers, layer, bidirectional, encoder_dim, num_embeddings, embedding_dim, padding_idx,
                 dropout, mlp_dropout, mlp_hidden_size, use_cuda=True,
                n_tasks=4):
        super(RNN_network, self).__init__()
        self.n_layers = n_layers
        self.layer = layer
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.encoder_dim = encoder_dim
        self.Embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.Embedding.weight)
        if layer == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, encoder_dim,
                               n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout, 
                               batch_first=True
                               #bias=False,
                              )
        elif layer == 'GRU':
            self.rnn = nn.GRU(embedding_dim, encoder_dim,
                              n_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout)
        else:
            self.layer = nn.RNN(embedding_dim, encoder_dim,
                                n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                batch_first=True) 
        self.mlp = nn.Linear(in_features=encoder_dim,
                                out_features=mlp_hidden_size)
        self.classifier = nn.Linear(in_features=mlp_hidden_size,
                             out_features=n_tasks*2)
        self.use_cuda = use_cuda
        self.n_tasks = n_tasks

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim, requires_grad=True).cuda()
        else:
            return torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim, requires_grad=True)

    def init_cell(self, batch_size):
        if self.use_cuda:
            return torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim, requires_grad=True).cuda()
        else:
            return torch.zeros(self.n_layers * self.n_directions,
                                            batch_size,
                                            self.encoder_dim, requires_grad=True)

            
    def forward(self, inp, eval=False, pack=True, previous_hidden=None):
        if eval:
            self.eval()
        else:
            self.train()
        input_tensor = inp[0]
        input_length = inp[1]
        batch_size = input_tensor.size(0)
        input_tensor = self.Embedding(input_tensor)
    
        if pack:
            input_lengths_sorted, perm_idx = torch.sort(input_length, dim=0, descending=True)
            input_lengths_sorted = input_lengths_sorted.detach().to(device).tolist()
            input_tensor = torch.index_select(input_tensor, 0, perm_idx)
            rnn_input = pack_padded_sequence(input=input_tensor,
                                             lengths=input_lengths_sorted,
                                             batch_first=True)
        else:
            rnn_input = input_tensor
        if previous_hidden is None:
            previous_hidden = self.init_hidden(batch_size)
            if self.layer == 'LSTM':
                cell = self.init_cell(batch_size)
                previous_hidden = (previous_hidden, cell)
        else:
            if self.layer == 'LSTM':
                hidden = previous_hidden[0]
                cell = previous_hidden[1]
                hidden = torch.index_select(hidden, 1, perm_idx)
                cell = torch.index_select(cell, 1, perm_idx)
                previous_hidden = (hidden, cell)
            else:
                previous_hidden = torch.index_select(previous_hidden, 1, perm_idx)
        rnn_output, next_hidden = self.rnn(rnn_input)  # , previous_hidden)

        if pack:
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            rnn_output = torch.index_select(rnn_output, 0, unperm_idx)
            if self.layer == 'LSTM':
                hidden = next_hidden[0]
                cell = next_hidden[1]
                hidden = torch.index_select(hidden, 1, unperm_idx)
                cell = torch.index_select(cell, 1, unperm_idx)
                next_hidden = (hidden, cell)
            else:
                next_hidden = torch.index_select(next_hidden, 1, unperm_idx)

        index_t = (input_length - 1).to(dtype=torch.long)
        index_t = index_t.view(-1, 1, 1).expand(-1, 1, rnn_output.size(2))

        embedded = torch.gather(rnn_output, dim=1, index=index_t).squeeze(1)
        
        #output, _ = self.rnn(embedded, previous_hidden)
        #embedded = output[-1, :, :].squeeze(0)
        output = embedded
        output = self.mlp(output)
        output = F.relu(output)
        output = self.classifier(output)
        # output = torch.sigmoid(output)
        output = output.view(-1, 2, self.n_tasks)

        if torch.isnan(output).any():
            sys.exit()
        return output

    @staticmethod
    def cast_inputs(sample, use_cuda=True, for_prediction=False):
        batch_mols = sample['tokenized_smiles'].to(dtype=torch.long)
        if for_prediction and "object" in sample.keys():
            batch_object = sample['object']
        else:
            batch_object = None  #always to to this line
        batch_length = sample['length'].to(dtype=torch.long)
        if not for_prediction and "labels" in sample.keys():
            # batch_labels = sample['labels'].to(dtype=torch.float)
            batch_labels = sample['labels'].to(dtype=torch.long)
            #if task == 'classification':
            #    batch_labels = batch_labels.to(dtype=torch.long)
        else:
            batch_labels = None
        if use_cuda:
            batch_mols = batch_mols.to(device)
            batch_length = batch_length.to(device)
            if batch_labels is not None:
                batch_labels = batch_labels.to(device)
        if batch_object is not None:
            return (batch_mols, batch_length), batch_object
        else:
            return (batch_mols, batch_length), batch_labels



def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def calculate_metrics(predicted, ground_truth, metrics):
    return metrics(ground_truth, predicted)

def train_step(model, optimizer, criterion, inp, target):
    optimizer.zero_grad()
    output = model.forward(inp, eval=False)
    # if torch.isnan(output).any():
    #     print(inp)
    #     print(target)
    #     sys.exit()
    # print(output, target)
    loss = criterion(output, target)
    if torch.isnan(loss).any():
        sys.exit()
    loss.backward()

    #clip the gradient to avoid vanishing gradients
    max_norm = 1.0
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    use_clip_grad = True
    max_grad_norm = 10.0
    if use_clip_grad:
        clip_grad_norm_(model.parameters(), max_grad_norm)

    return loss

def fit(model, scheduler, train_loader, optimizer, criterion, logdir,
        print_every, 
        save_every, 
        n_epochs, 
        eval=False, val_loader=None):
    cur_epoch = 0
    start = time.time()
    loss_total = 0
    n_batches = 0
    all_losses = []
    val_losses = []   
    world_size = 1
    for epoch in trange(cur_epoch, n_epochs + cur_epoch):
        for i_batch, sample_batched in enumerate(train_loader):
            batch_input, batch_target = model.cast_inputs(sample_batched)
            loss = train_step(model, optimizer, criterion,
                              batch_input, batch_target)
            if world_size > 1:
                reduced_loss = reduce_tensor(loss, world_size)
            else:
                reduced_loss = loss.clone()
            loss_total += reduced_loss.item()
            n_batches += 1
        cur_loss = loss_total / n_batches
        all_losses.append(cur_loss)

        if epoch % print_every == 0:
            print('TRAINING: [Time: %s, Epoch: %d, Progress: %d%%, '
                  'Loss: %.4f]' % (time_since(start), epoch,
                                   epoch / n_epochs * 100, cur_loss))
            if eval:
                assert val_loader is not None
                val_loss, metrics = evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), logdir + '/checkpoint/epoch_' + str(epoch))

        loss_total = 0
        n_batches = 0
        scheduler.step()

    return all_losses, val_losses

def multitask_f1(ground_truth, predicted, return_mean=True):
    from sklearn.metrics import roc_auc_score, f1_score
    import numpy as np
    import torch
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    predicted = np.array(predicted >= 0.5, dtype="int")
    n_tasks = ground_truth.shape[1]
    auc = []
    for i in range(n_tasks):
        ind = np.where(ground_truth[:, i] != 999)[0]
        auc.append(f1_score(ground_truth[ind, i], predicted[ind, i]))
    #if torch.distributed.get_rank() == 0:
    #    print(auc)
    if return_mean:
        return np.mean(auc)
    else:
        return auc


def multitask_auc(ground_truth, predicted, return_mean=True):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import torch
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    n_tasks = ground_truth.shape[1]
    auc = []
    for i in range(n_tasks):
        ind = np.where(ground_truth[:, i] != 999)[0]
        auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind, i]))
    #if torch.distributed.get_rank() == 0:
    #    print(auc)
    if return_mean:
        return np.mean(auc)
    else:
        return auc

def evaluate(model, val_loader, criterion):
    loss_total = 0
    n_batches = 0
    start = time.time()
    prediction = []
    ground_truth = []
    has_module = False
    eval_metrics = multitask_auc
    world_size = 1
    for i_batch, sample_batched in enumerate(val_loader):
        batch_input, batch_target = model.cast_inputs(sample_batched)
        predicted = model.forward(batch_input, eval=True)
        prediction += list(predicted.detach().cpu().numpy())
        ground_truth += list(batch_target.cpu().numpy())
        loss = criterion(predicted, batch_target)
        loss_total += loss.item()
        n_batches += 1

    cur_loss = loss_total / n_batches
    prediction2 = [proba[1, :] for proba in prediction]
    metrics = calculate_metrics(prediction2, ground_truth,
                                eval_metrics)  
    print('EVALUATION: [Time: %s, Loss: %.4f, Metrics: %.4f]' %
          (time_since(start), cur_loss, metrics))
    return cur_loss, metrics



# class MultitaskLoss(_WeightedLoss):
#     r"""
#     Creates a criterion that calculated binary cross-entropy loss over
#     `n_tasks` tasks given input tensors `input` and `target`. Returns loss
#     averaged across number of samples in every task and across `n_tasks`.
#     It is useful when training a classification model with `n_tasks`
#     separate binary classes.

#     The loss can be described as:

#     ..math::
#         \text{loss}(y, t) = -\frac{1}{n_tasks}\sum_{i=1}^{n_tasks}\frac{1}{N_i}
#         \sum_{j=1}^{N_i} \left(t[i, j]\log(1-y[i, j])
#         + (1-t[i, j])\log(1-y[i, j])\right).

#     Args:
#         ignore_index (int): specifies a target value that is ignored
#             and does not contribute to the gradient. For every task losses are
#             averaged only across non-ignored targets.
#         n_tasks (int): specifies number of tasks.

#     Shape:
#         -Input: :math: `(N, n_tasks)`. Values should be in :math:`[0, 1]` range,
#             corresponding to probability of class :math:'1'.
#         -Target: :math: `(N, n_tasks)`. Values should be binary: either
#             :math:`0` or :math:`1`, corresponding to class labels.
#         -Output: scalar.

#     """

#     def __init__(self, ignore_index, n_tasks):
#         super(MultitaskLoss, self).__init__(reduction='none')
#         self.n_tasks = n_tasks
#         self.ignore_index = ignore_index

#     def forward(self, input, target):
#         if torch.isnan(input).any():
#             print(input)
#             print(target)
#             sys.exit
#         assert target.size()[1] == self.n_tasks
#         assert input.size()[1] == self.n_tasks
#         x = torch.zeros(target.size()).cuda()
#         y = torch.ones(target.size()).cuda()
#         mask = torch.where(target == self.ignore_index, x, y)
#         loss = F.binary_cross_entropy(input, mask*target,
#                                       weight=self.weight)
#         loss = loss*mask
#         n_samples = mask.sum(dim=0)  #bug! this is the sum over the rows! not the number of the samples. So it could contain 0, which causes nan values when dividing
#         return (loss.sum(dim=0)/n_samples).mean() 


def create_loader(dataset, batch_size, shuffle=True, num_workers=1,
                  pin_memory=False, sampler=None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory, sampler=sampler)
    return data_loader


if __name__ == "__main__":
    tasks = ["P23458","O60674","P52333","P29597"]
    n_tasks = len(tasks)
    tokens = pickle.load(open('tokens.pkl', 'rb'))
    train_dataset = SmilesDataset("data/train0.csv", smiles_col='StdSMILES', target_cols=tasks, tokens=tokens)#["P23458","O60674","P52333","P29597"])

    # pickle.dump(train_dataset.tokens, open('tokens.pkl', 'wb'))

    train_sampler = None
    train_loader = create_loader(train_dataset,
                            batch_size=64,
                            shuffle=(train_sampler is None),
                            num_workers=10,
                            pin_memory=True,
                            sampler=train_sampler)

    
    val_dataset = SmilesDataset("data/val0.csv", smiles_col='StdSMILES', target_cols=tasks,#["P23458","O60674","P52333","P29597"], 
                                tokens=tokens)
    val_loader = create_loader(val_dataset,
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=6,
                                    pin_memory=True)


    model = RNN_network(n_layers=2,
                        layer="GRU", 
                        bidirectional=False,
                        embedding_dim=256,
                        encoder_dim=256,
                        num_embeddings=len(tokens),
                        padding_idx=tokens.index(" "),
                        dropout=0.8,
                        mlp_dropout=0.8,
                        mlp_hidden_size=256
                    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # criterion = MultitaskLoss(ignore_index=999, n_tasks=n_tasks)
    criterion = CrossEntropyLoss(ignore_index=999)

    logdir="lstm/"
    import os
    try:
        os.stat(logdir)
    except:
        os.mkdir(logdir)
        print('Directory created')
    ckpt_dir = logdir + '/checkpoint/'
    try:
        os.stat(ckpt_dir)
    except:
        os.mkdir(logdir + '/checkpoint')
        print("Directory created")

    
    all_losses, val_losses = fit(model, lr_scheduler, train_loader, optimizer, criterion,
                                logdir=logdir,
                                print_every=1,
                                save_every=5,
                                n_epochs=100,
                                eval=True, val_loader=val_loader)

    pickle.dump([all_losses, val_losses], open('lstm_training_metrics.pkl', 'wb'))

