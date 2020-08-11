#this is the maximum number of tokens in the sentence 
#batch sizes is small because model is huge! 
#define path to BERT model files 
#define the tokenizer  
#used tokenizer and model 
#from huggingface's transformers 

import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 12
EPOCHS = 8
BERT_PATH = "../input/bert-base-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/absa-dataset/input/bert-pair/train_NLI_M.tsv"
DEV_FILE = "../input/absa-dataset/input/bert-pair/dev_NLI_M.tsv"
TEST_FILE = "../input/absa-dataset/input/bert-pair/test_NLI_M.tsv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
