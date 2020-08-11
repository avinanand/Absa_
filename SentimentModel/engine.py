import torch
import torch.nn as nn
from tqdm import tqdm
import pdb


def loss_fn(outputs, targets):
    #targets = torch.argmax(targets, 1)
    #loss = nn.CrossEntropyLoss()(outputs, targets.view(-1, 1))
    loss = nn.CrossEntropyLoss()(outputs, targets)
    
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        #pdb.set_trace()
        targets = targets.to(device, dtype=torch.long)
        #pdb.set_trace()

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            act = nn.Softmax(dim=1)
            outputs = act(outputs)
            fin_targets.extend(targets.detach().cpu().numpy().tolist())
            fin_outputs.extend(outputs.cpu().numpy().tolist())
            
    return fin_outputs, fin_targets