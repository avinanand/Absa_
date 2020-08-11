import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import preprocessing

def run():
    #this function trains the model  
    #initialize BERTDataset from dataset.py 
    #for training dataset 
    #create training dataloader 
    #initialize the cuda device 
    #use cpu if you dont have GPU 
    #create parameters we want to optimize  
    #we generally dont use any decay for bias  
    #and weight layers 
    
    encoder = preprocessing.LabelEncoder()
    train.loc[:, "label"] = encoder.fit_transform(train["label"])
    #dev_df.loc[:, "label"] = encoder.transform(dev_df["label"])
    test.loc[:, "label"] = encoder.transform(test["label"])
    
    
    train_dataset = BERTDataset(sentence1s = train.text.values, targets=train.label.values)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4)
    
    valid_dataset = BERTDataset(sentence1s = test.text.values,targets=test.label.values)
    
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4)
    
    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001,},
                            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},]
    num_train_steps = int(len(train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    
    model = nn.DataParallel(model)
    
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(epoch)
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        #pdb.set_trace()
        outputs = torch.tensor(outputs)
        outputs = torch.argmax(outputs, dim=1)
        #outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score =", {accuracy})
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy
            
if __name__ == "__main__":
    run()
