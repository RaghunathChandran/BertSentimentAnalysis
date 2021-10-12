from re import U
from fastapi import FastAPI
from training.train import train
import uvicorn
from sentiment_prediction.sentiment_predict import PredictSentiment

app = FastAPI()
'''
@app.get("/Hello")
async def HelloWorld():
    return {"message": "Hello World"}

@app.get("/train")
async def training():
    train.main()
'''
@app.post("/predict")
async def predict(review: str, entity: str):
    #review = "I've already ordered my new iPhone and couldn't be more pleased with the service - I just wish I'd called them sooner instead of waiting for it to dry out in rice for 2 weeks without success." 
    #entity = "weeks"

    predict = PredictSentiment(review, entity)
    result = predict.predict()
    if(result == 1 ):
        return {"result": "Positive"}
    else:
        return {"result": "Negative"}

if __name__ == "__main__":
    if __debug__:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)