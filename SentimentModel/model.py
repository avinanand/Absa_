#we fetch the model from the BERT_PATH defined in 
#config.py 
#BERT in its default settings returns two outputs 
#last hidden state and output of bert pooler layer  
#used the output of the pooler which is of the size  
#(batch_size, hidden_size)  
#hidden size can be 768 or 1024 depending on  
#if we are using bert base or large respectively  
#in our case, it is 768  
#note that this model is pretty simple 
#might want to use last hidden state 
#or several hidden states 


import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 4)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
