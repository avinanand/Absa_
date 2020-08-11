import torch

class BERTDataset:
    def __init__(self, sentence1s,targets):
        self.sentence1s = sentence1s
        #self.sentence2s = sentence2s
        self.targets = targets
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.sentence1s)
                   
        
    def __getitem__(self, item):
        sentence1 = str(self.sentence1s[item])
        sentence1 = " ".join(sentence1.split())
        
        #sentence2 = str(self.sentence2s[item])
        #sentence2 = " ".join(sentence2.split())

        inputs = self.tokenizer.encode_plus(sentence1,
                                            None,
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
