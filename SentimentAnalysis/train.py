import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from packaging import version
from preprocessing.preprocess_sentiment_data import TextPreprocessing
from bert import bert_sentiment_tokenizer
from sklearn.model_selection import StratifiedKFold
from bert.bert_sentiment import BertSentiment
from bert.bert_sentiment_tokenizer import BertSentimentTokenizer


LOG_PATH = os.getcwd() + '\logs\scalars'

seq_len = 512
batch_size = 16
EPOCHS = 500
kfold_splits = 10

def map_func(input_ids, masks, labels):

    # Convert three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

def get_model_name(k):
    return '\model_'+str(k)

def main():
  
    skf = StratifiedKFold(n_splits = kfold_splits, random_state = 7, shuffle = True) 
    working_directory = os.getcwd()

    save_dir = os.getcwd()+'\saved_models'
    fold_var = 1
    preprocessed_data = TextPreprocessing('\Entity_sentiment_trainV2.xlsx').preprocess_data()
    tokenizer = BertSentimentTokenizer('bert-base-cased')

    Y = preprocessed_data[['Sentiment']]
    X = preprocessed_data['Sentence'] 

    for train_index, val_index in skf.split(X,Y):
        training_data = X.iloc[train_index]
        validation_data = X.iloc[val_index]
    
        #-------------------------------------------------Train DataSet ----------------------------------------------------#
   
        train_lbl =  Y.iloc[train_index]    
        val_lbl =  Y.iloc[val_index]

 
        tokenized_train_dataset = tokenizer.tokenize_data(data = training_data, label = train_lbl)
        tokenized_train_dataset = tokenized_train_dataset.batch(batch_size, drop_remainder=True)

        tokenized_validation_dataset = tokenizer.tokenize_data(data = validation_data, label = val_lbl)
        tokenized_validation_dataset = tokenized_validation_dataset.batch(batch_size, drop_remainder=True)

        logdir = str.format('{0}\{1}', LOG_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
        #-------------------------------------------------Validation DataSet ----------------------------------------------------#
     
        bert = BertSentiment('bert-base-cased')
    
        checkpoint_filepath = save_dir + get_model_name(fold_var)
        
        if os.path.exists(save_dir + get_model_name(10)):
            #Load weights from saved model run earlier for each time
            bert.model.load_weights(save_dir + get_model_name(10))
        
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=False,
                                                                   monitor='val_accuracy',
                                                                   mode='max',
                                                                   verbose=1,
                                                                   save_best_only=True)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=5)

        # Model weights are saved at the end of every epoch, if it's the best seen
        # so far.
        bert.model.fit(tokenized_train_dataset, 
              validation_data=tokenized_validation_dataset,
              epochs=EPOCHS, 
              callbacks=[model_checkpoint_callback, tensorboard_callback, earlyStopping], 
              verbose=1)

        # The model weights (that are considered the best) are loaded into the model.
        bert.model.load_weights(save_dir +"\model_"+str(fold_var))
    
        fold_var +=1
        tf.keras.backend.clear_session()

def train():
    main()

if __name__ == "__main__":
    main()
