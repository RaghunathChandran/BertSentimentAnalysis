import pandas as pd
import unicodedata
import os

DATA_PATH = os.getcwd() + "\data"

'''
Pre processign text data.
'''

class TextPreprocessing():


    def __init__(self, file_name: str=None):
        self.file_name = file_name

        if self.file_name != None:
            self.dataset = pd.read_excel(DATA_PATH + file_name)

    def preprocess_data(self, isPredict:bool = None, sentence: str=None, entity: str=None):        
        if isPredict:
            return self.__process_prediction_dataset()

        elif sentence != None and entity != None:
            return self.__process_prediction_data(sentence, entity)

        df = pd.DataFrame(columns=['Sentence', 'Sentiment'])
        df['Sentiment'] = self.sentiment_mapping()
        df['Sentence'] = self.create_targetted_text()
        return df

    def __process_prediction_dataset(self):
        df = pd.DataFrame(columns=['Sentence'])
        df['Sentence'] = self.create_targetted_text()
        return df

    def __process_prediction_data(self, sentence, entity):
        self.sentence = sentence
        self.entity = entity

    def sentiment_mapping(self):        
        df = self.dataset
        binary_mapping = {'negative': 0, 'positive': 1}
        df['Sentiment'] = df['Sentiment'].map(binary_mapping)
        return df['Sentiment']
 
    def create_targetted_text(self, sentence: str=None, entity: str=None):
        
        # VERY LOUSY way of linking Entity with sentence : TO DO -create a proper pipeline for training
        if sentence != None and entity != None:
            return sentence + ' | ' + entity
        else:
            df = self.dataset            
            df['Sentence'] = self.dataset['Sentence'] + ' | ' + self.dataset['Entity'] 
            return df['Sentence']



     

