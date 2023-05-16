import torch

torch.manual_seed(123)

#prepare sample predictions and targets
orig_pred = torch.rand((3, 2, 4))
pred = torch.softmax(orig_pred, dim=1)

target = torch.randint(0, 2, (3, 4))

target[0, 0] = 999
target[2, 2] = 999

mask = (target <= 1.)

# method 1: using CrossEntropyLoss to calculate loss
loss = torch.nn.CrossEntropyLoss(ignore_index=999)

print('method 1', loss(orig_pred, target))

# method 2: using CrossEntropyLoss to calculate loss with mask
loss = torch.nn.CrossEntropyLoss(reduction="none")

q = pred[0, 0, 1]
p = pred[0, 1, 1]
t = target[0, 1]
# print(- t*torch.log(p) - (1-t)*torch.log(q))


# print(loss(orig_pred, target*mask)*mask)
print('method 2', torch.sum(loss(orig_pred, target*mask)*mask)/mask.sum())


# method 3: using BCELoss to calculte loss with mask
pred = pred[:, 1, :]

# print(pred.shape)

loss = torch.nn.BCELoss(reduction="none")

p = pred[0, 1]
t = target[0, 1]
# print(- t*torch.log(p) - (1-t)*torch.log(1-p))

# print(loss(pred, target.float()*mask)*mask)
print('method 3', torch.sum(loss(pred, target.float())*mask)/mask.sum())

