#we fetch max len and tokenizer from config.py  self.tokenizer = config.TOKENIZER  self.max_len = config.MAX_LEN 
#encode_plus comes from hugginface's transformers 
#and exists for all tokenizers they offer  
#it can be used to convert a given string  #to ids, mask and token type ids which are 
#needed for models like BERT   


import config
import torch

class BERTDataset:
    def __init__(self, sentence1s, sentence2s, targets):
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.sentence1s)
                   
        
    def __getitem__(self, item):
        sentence1 = str(self.sentence1s[item])
        sentence1 = " ".join(sentence1.split())

        sentence2 = str(self.sentence2s[item])
        sentence1 = " ".join(sentence2.split())

        inputs = self.tokenizer.encode_plus(sentence1,
                                            sentence2, 
                                            add_special_tokens=True, 
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                           )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[item], dtype=torch.long),
        }
