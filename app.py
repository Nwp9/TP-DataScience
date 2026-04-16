from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, insert

engine = create_engine("sqlite:///house_prices.db")
metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("neighborhood", String),
    Column("gr_liv_area", Float),
    Column("overall_qual", Integer),
    Column("garage_cars", Integer),
    Column("predicted_price", Float),
)

metadata.create_all(engine)


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Initialisation de l'API
app = FastAPI()

# Chargement du modèle
model = joblib.load("ridge_house_price_model.joblib")

# Schéma des données d'entrée
class HouseInput(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: int
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: int
    BsmtFinType2: str
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    FirstFlrSF: int
    SecondFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: int
    GarageFinish: str
    GarageCars: int
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: str
    Fence: str
    MiscFeature: str
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

# Route test
@app.post("/predict")
def predict_price(house: HouseInput):
    data = house.model_dump()

    # rename
    data["1stFlrSF"] = data.pop("FirstFlrSF")
    data["2ndFlrSF"] = data.pop("SecondFlrSF")
    data["3SsnPorch"] = data.pop("ThreeSsnPorch")

    df = pd.DataFrame([data])

    # alignement
    df = df.reindex(columns=model.feature_names_in_)

    # prédiction
    pred_log = model.predict(df)[0]
    pred_price = float(np.expm1(pred_log))

    # sauvegarde en base
    conn = engine.connect()

    stmt = insert(predictions).values(
        neighborhood=data["Neighborhood"],
        gr_liv_area=data["GrLivArea"],
        overall_qual=data["OverallQual"],
        garage_cars=data["GarageCars"],
        predicted_price=pred_price
    )

    conn.execute(stmt)
    conn.commit()

    return {
        "pred_log": round(float(pred_log), 4),
        "pred_price": round(pred_price, 2)
    }

from sqlalchemy import select

@app.get("/history")
def get_history():
    conn = engine.connect()
    query = select(predictions)
    result = conn.execute(query)

    rows = [dict(row._mapping) for row in result]
    return rows