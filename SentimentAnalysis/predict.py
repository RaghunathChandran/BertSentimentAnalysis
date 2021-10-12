import os
from os import path
from sentiment_prediction.sentiment_predict import PredictSentiment


path = os.getcwd() + "\data\Entity_sentiment_testV2.xlsx"
file_name = "\Entity_sentiment_testV2.xlsx"

def main():
    predict = PredictSentiment('','')
    results = predict.predict_from_file(filepath=file_name)
    return results


if __name__ == "__main__":
    main()