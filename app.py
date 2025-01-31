import sys
import os
import pandas as pd

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging
from fooddelivery.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from fooddelivery.utils.main_utils.utils import load_object

from fooddelivery.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from fooddelivery.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from fooddelivery.utils.ml_metrics.model.estimator import NetworkModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# FastApi setup
app = FastAPI()
origins = ["*"]

# to make surre we will be able to access it in the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"], summary="Redirect to API docs")
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["ML Operations"], summary="Run training pipeline")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response(content="Training is Successful.", status_code=200)
    except Exception as e:
        logging.error(f"Error during training{e}")
        raise FoodDeliveryException(e,sys)
    
@app.post("/predict", tags=["ML Operations"], summary="Make predictions")
async def predict_route(request:Request, file:UploadFile=File(...)):
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)

        # load models
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        print(df.iloc[0])

        # Make Prediction
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])

        # save the prediction in a csv file
        df.to_csv("prediction_output/output.csv")

        #convert dataframe to HTML table
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html", {"request":request, "table":table_html})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise FoodDeliveryException(e,sys)

if __name__ == "__main__":
    app_run(app,host="0.0.0.0",port=8000)
