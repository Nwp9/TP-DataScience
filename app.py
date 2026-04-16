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
@app.get("/")
def home():
    return {"message": "API House Price prête"}

# Route prédiction
@app.post("/predict")
def predict_price(house: HouseInput):
    data = house.model_dump()

    # rename
    data["1stFlrSF"] = data.pop("FirstFlrSF")
    data["2ndFlrSF"] = data.pop("SecondFlrSF")
    data["3SsnPorch"] = data.pop("ThreeSsnPorch")

    df = pd.DataFrame([data])

    # alignement exact
    df = df.reindex(columns=model.feature_names_in_)

    pred_log = model.predict(df)[0]
    pred_price = float(np.expm1(pred_log))

    return {
        "pred_log": round(float(pred_log), 4),
        "pred_price": round(pred_price, 2)
    }