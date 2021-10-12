import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from datetime import datetime
from utils import utils


LOG_PATH = os.getcwd() + '\logs\scalars'

logdir = str.format('{0}\{1}', LOG_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
seq_len = 512
batch_size = 16
EPOCHS = 5

# This is a Singleton class 
class BertSentimentTokenizer(object):

    _instance = None

    def __new__(self, tokenizer_path: str):
        if self._instance is None:
            print('Creating the object')
            self._instance = super(BertSentimentTokenizer, self).__new__(self)
            ##cls.model = tf.keras.models.load_model(SAVED_MODEL)
            # Put any initialization here.
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return self._instance

    #def __init__(self, tokenizer_path: str):
    #    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    def __tokenize_training_data(self, data, label):
        num_samples = len(data)
        label_arr = np.array(label['Sentiment'].values.tolist())

        data_labels = np.zeros((num_samples, label_arr.max()+1))

    
        data_labels[np.arange(num_samples), label_arr] = 1

        
        data_tokens = self.tokenizer(data.tolist(), max_length=seq_len, truncation=True,
                                padding='max_length', add_special_tokens=True,
                                return_tensors='np') 
    
        tokenized_dataset = tf.data.Dataset.from_tensor_slices((data_tokens['input_ids'], 
                                                        data_tokens['attention_mask'], 
                                                        data_labels))
                                                         #return {'input_ids': input_ids, 'attention_mask': masks}, labels
        tokenized_dataset = tokenized_dataset.map(self.map_func)
 
        if __debug__:            
            print(tokenized_dataset.take(1))
        return  tokenized_dataset

    
    def __tokenize_prediction_data(self, text: str):

        #str ""
        data_tokens = self.tokenizer.encode_plus(text, max_length=512, truncation=True,
                                padding='max_length', add_special_tokens=True,
                                return_tensors='tf')

        return {'input_ids': data_tokens['input_ids'],
            'attention_mask': data_tokens['attention_mask']}

    def tokenize_data(self, data=None, label=None, text:str=None):
        # initialize tokenizer
        #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        #tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
        if text ==None:
            return self.__tokenize_training_data(data, label)
        if data == None:
            return self.__tokenize_prediction_data(text)

    def map_func(self, input_ids, masks, labels):
        # Convert  three-item tuple into a two-item tuple where the input item is a dictionary
        return {'input_ids': input_ids, 'attention_mask': masks}, labels


    
