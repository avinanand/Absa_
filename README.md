# Absa_

# OutPut files are 

# 1.predicted_aspect(1).json 

# 2.sentiment_prediction.json

Divided both prediction to get a better view for models output

Two models are trained 

  1.For aspect prediction
  
  2.For sentiment prediction
  
 Ipynb file predict_aspect.ipynb file for aspect prediction
 
 confi.py: This python file has all the necessary configuration
 
 
 dataset.py : In this i created the dataset required for training 
 
 
 model.py: we create the model for training in this file
         we fetch the model from the BERT_PATH defined in
         
         
 engine.py: we have loss function adn two other one for trainin and another for evaluation
              In Training function:
                InThis is the training function which trains for one epoch 
                :param data_loader: it is the torch dataloader object  
                :param model: torch model bert in our case 
                :param optimizer: adam, sgd, etc 
                :param device: can be cpu or cuda  
                :param scheduler: learning rate scheduler 
              In eval funcion :
                this is the validation function that generates  predictions on validation data 
                :param data_loader: it is the torch dataloader object 
                :param model: torch model, bert in our case  
                :param device: can be cpu or cuda 
                :return: output and targets 

train.py 
 #this function trains the model  
 #read the training file and fill NaN values with "none" 
 #you can also choose to drop NaN values in this 
 #specific dataset 

 

            
