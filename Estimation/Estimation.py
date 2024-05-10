from joblib import load
import pandas as pd
models_list={
    "ODAS":'models/ridge_model_odas_10May2024.joblib',
    "THYAO":'models/ridge_model_thyao_10May2024.joblib',
    "ASELS":'models/ridge_model_asels_10May2024.joblib'
}
window_size=42

class Estimator:
    def __init__(self,share="ODAS"):
        self.__model = load(models_list[share])
    def predict(self,datas,window_size):
        if len(datas)!=window_size:
            print("veri boyutu hatası")
        else:
            predictions=self.__model.predict(datas.reshape(1,-1))
            return predictions

"""
Test Kodu:
df datasets klasöründeki veri setlerinden window_size kadar veri alır.
Estimator içindeki share'e hisse adı girilir tahmin yaptırılır

"""
df = pd.read_excel('datasets/thyao.xlsx')
result=Estimator(share="THYAO").predict(window_size=window_size,datas=df["Kapanış(TL)"].iloc[-window_size:].to_numpy())
print("Result")
