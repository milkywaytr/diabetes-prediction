from fastapi import FastAPI
import uvicorn
import pickle
from models import Women


app=FastAPI()

model=pickle.load(open("model.pkl","rb"))
@app.get("/{name}")
def hello(name):
    return {"Hello {} and welcome to this API".format(name)}

@app.get("/")
def greet():
    return {"Hello World!"}

@app.post("/predict")
def predict(req:Women):
    preg=req.pregnancies
    glucose=req.glucose
    bp=req.bp
    skinthickness=req.skinthickness
    insulin=req.insulin
    bmi=req.bmi
    dpf=req.dpf
    age=req.age
    
    features=list([preg,glucose,bp,skinthickness,insulin,bmi,dpf,age])
    predict=model.predict([features])
    probab=model.predict_proba([features])
    
    if(predict==1):
        return {"ans":"You have been tested positive with {} probability".format(probab[0][1])}
    else:
        return {"ans":"You have been tested negative with {} probability".format(probab[0][0])}

 

if __name__=="__main__":
    uvicorn.run(app)