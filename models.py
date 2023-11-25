from pydantic import BaseModel

class Women(BaseModel):
    pregnancies:int
    glucose:int
    bp:int
    skinthickness:int
    insulin:int
    bmi:float
    dpf:float
    age:int