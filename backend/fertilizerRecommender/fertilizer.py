import os
import json
import joblib
import numpy as np
import pandas as pd

class FertilizerRecommender:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))  
        
        labels_path = os.path.join(base_dir, "labels.json")
        col_trans_path = os.path.join(base_dir, "colTrans.jbl.lzma")
        model_path = os.path.join(base_dir, "model.jbl.lzma")

        with open(labels_path) as f:
            self.labels = json.load(f)
        self.ct = joblib.load(col_trans_path)
        self.model = joblib.load(model_path)
    def predict(self, data):
        data = pd.DataFrame([data])  
        data.columns = [
            "Temparature",
            "Humidity",
            "Moisture",
            "Soil Type",
            "Crop Type"
        ]

        data["Soil_Code"] = data["Soil Type"].map(self.labels[0])
        data["Crop_Code"] = data["Crop Type"].map(self.labels[1])

        data.drop(["Soil Type", "Crop Type"], axis=1, inplace=True)

        fertMap = {v: k for k, v in self.labels[2].items()}

        fertilizer = self.model.predict(self.ct.transform(data))[0]

        return fertMap[fertilizer]

