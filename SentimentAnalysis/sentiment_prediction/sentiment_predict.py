import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from preprocessing.preprocess_sentiment_data import TextPreprocessing

from bert.bert_sentiment import BertSentiment
from bert.bert_sentiment_tokenizer import BertSentimentTokenizer
from sentiment_prediction.LoadSavedModel import LoadSavedModelSingleton

SAVED_MODEL = os.getcwd()+'\saved_models\model_10'
RESULT_DIR = os.getcwd()+'\\results'
RESULT_PATH = str.format('{0}\{1}', RESULT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

class PredictSentiment:

    def __init__(self, sentence: str, entity: str):
        self.sentence = sentence
        self.entity = entity
    
    def predict_from_file(self, filepath: str):
        model = tf.keras.models.load_model(SAVED_MODEL)
   
        prediction_data = TextPreprocessing(filepath)
        prediction_data_copy = prediction_data.dataset.copy()
        prediction_df = prediction_data.preprocess_data(True)

        tokenizer = BertSentimentTokenizer('bert-base-cased')

        prediction_df['Sentiment'] = None

        for i, row in prediction_df.iterrows():
            tokenized_data = tokenizer.tokenize_data(text=row['Sentence'])
            probs = model.predict(tokenized_data)[0]
            prediction = np.argmax(probs)
            prediction_df.at[i,'Sentiment'] = prediction

        prediction_df.head()   
        prediction_data_copy['Sentiment'] = prediction_df['Sentiment']

        RESULT_PATH = str.format('{0}\sentiment_results{1}.xlsx', RESULT_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))
        prediction_data_copy.to_excel(RESULT_PATH, index = False, header=True)
        print(str.format("File sentiment_results{0}.xlsx save at {0}", datetime.now().strftime("%Y%m%d%H%M%S"), RESULT_DIR))

        tf.keras.backend.clear_session() 
        
    def predict(self):
        #model = tf.keras.models.load_model(SAVED_MODEL)
        #model  = LoadSavedModelSingleton.getInstance()
        model = LoadSavedModelSingleton().model
        
        preprocesssed_data = TextPreprocessing()
        prepared_data = preprocesssed_data.create_targetted_text(sentence=self.sentence, entity=self.entity)
        tokenizer = BertSentimentTokenizer('bert-base-cased')

        tokenized_data = tokenizer.tokenize_data(text=prepared_data)

        probs = model.predict(tokenized_data)[0]

        result = np.argmax(probs)    
        tf.keras.backend.clear_session()    
        return result
