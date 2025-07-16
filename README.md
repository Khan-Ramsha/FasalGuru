# AgriAssist: Real-Time Agricultural Decision Support Platform

**AgriAssist** is a modular, API-based platform using AI models to assist farmers and agri-businesses in:
- Crop disease/pest detection (from leaf images)
- Fertilizer recommendation
- Real-time irrigation
- Agri-commodity price prediction

## Features
- **Image-Based Disease/Pest Diagnosis:** Upload a crop image and receive likely disease/pest class with prevention suggestions tailored to real-time weather.
- **Smart Fertilizer Recommender:** Real-time weather passed along with user input for soil and crop type to get a best-fit fertilizer suggestion.
- **Water Requirement (Irrigation Advisor):** Recommends irrigation quantity based on recent weather and basic soil/crop data.
- **Food Price Predictor:** Predicts expected prices for selected commodities.

## Technical Stack
- **Backend**: FastAPI, Python
- **ML/DL Libraries**: PyTorch (for image models), scikit-learn (tabular models)
- **Serving**: Uvicorn, REST API endpoints
- **Data Sources**: Kaggle

## Future Plans (Coming soon)
- Replacing use of current Resnet Model with using Pretrained Vision Transformers (ViT) with Transfer Learning
- Finetuning Models on Huge Data for better performance
- Image Validation for rejecting non-agri related images
- Adding ensemble methods and applying Data Augmentation techniques to improve Generalization
- Switching Local Ollama to cloud-based LLM call

## Demo
https://github.com/user-attachments/assets/18c6cec3-7d3b-4e20-9732-6d60312a95b6

Contributions are welcome! If you find some bugs or have any suggestions, feel free to drop your Pull Requests!!
