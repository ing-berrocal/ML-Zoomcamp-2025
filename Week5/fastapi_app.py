from fastapi import FastAPI, Path, Query
from pydantic import BaseModel 
from typing import Optional
import uvicorn

import pickle

app = FastAPI(title="MLZoomcamp", version="0.1")

with open('./model/pipeline_v1.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    pipeline = pickle.load(f_in)

###
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
###

# Pydantic model para recibir datos JSON
class CustomerData(BaseModel):
    lead_source: Optional[str] = None
    number_of_courses_viewed: Optional[int] = None
    annual_income: Optional[float] = None

def predict_test(record) :
    churn = pipeline.predict_proba(record)[0, 1]
    return round(churn,3)
     
@app.post("/predict", status_code=201)
def test_churn(customer_data: CustomerData):
    # Convertir el objeto Pydantic a diccionario para el modelo
    customer_dict = customer_data.dict()
    
    # En una app real aquí guardarías en la base de datos
    churn = predict_test(customer_dict)
    return {
        "churn_probaility":churn,
        "churn": bool(churn >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)