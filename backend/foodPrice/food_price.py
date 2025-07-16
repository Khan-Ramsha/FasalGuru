
import joblib
import numpy as np
class FoodPricePredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        
    def predict(self, input_values: list):
        print(self.model)
        input_array = np.array(input_values).reshape(1, -1)
        prediction = self.model.predict(input_array)
        return prediction[0][0]
