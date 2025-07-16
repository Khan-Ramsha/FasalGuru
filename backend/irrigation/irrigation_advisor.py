import os
import json
import joblib
import pandas as pd

class WaterRequirementPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_dir, "irrigation_model.pkl")
        labels_path = os.path.join(base_dir, "labels.json")

        self.model = joblib.load(model_path)

        with open(labels_path, "r") as f:
            self.labels = json.load(f)
            
    def preprocess_input(self, crop_type, soil_type, temperature, humidity, soil_moisture, rainfall):
        """
        Convert categorical inputs into numerical values using label encoding.
        """
        try:
            crop_numeric = self.labels["Crop Type"].get(crop_type)
            soil_numeric = self.labels["Soil Type"].get(soil_type)

            if crop_numeric is None or soil_numeric is None:
                raise ValueError("Invalid Crop Type or Soil Type!")

            # Create input DataFrame
            data = pd.DataFrame([[crop_numeric, soil_numeric, temperature, humidity, soil_moisture, rainfall]],
                                columns=["Crop Type", "Soil Type", "Temperature (°C)", "Humidity (%)", "Soil Moisture (%)", "Rainfall (mm)"])
            
            return data
        except Exception as e:
            raise ValueError(f"Error in preprocessing: {str(e)}")
        
    def predict(self, crop_type, soil_type, temperature, humidity, soil_moisture, rainfall):
        """
        Predict water requirement using the trained model.
        """
        try:
            input_data = self.preprocess_input(crop_type, soil_type, temperature, humidity, soil_moisture, rainfall)
            prediction = self.model.predict(input_data)[0]  
            return {"water_requirement": prediction}

        except Exception as e:
            return {"error": str(e)}