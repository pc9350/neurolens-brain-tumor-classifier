# NeuroLens: AI-Powered Brain Tumor Analysis

NeuroLens is an advanced web application designed to assist medical professionals in the analysis and classification of brain tumors from MRI scans. Leveraging state-of-the-art deep learning models and explainable AI techniques, NeuroLens provides detailed insights into brain tumor classification with visual explanations of focus areas.

🔗 Live Demo (Hugging Face Space):
https://pranavch-neurolens-brain-tumor.hf.space/

📦 Model Weights:
Hosted on Hugging Face and loaded at runtime.
This repository contains code only.

## Features

- **Multi-class Tumor Classification**: Classifies brain MRIs into Glioma, Meningioma, Pituitary, or No Tumor categories
- **Model Options**: Choose between Transfer Learning (Xception) or Custom CNN models
- **Explainable AI**: Visualize model decision areas with gradient-based saliency maps
- **AI-Generated Medical Explanations**: Receive expert-like explanations of findings powered by Google's Gemini API
- **Downloadable Reports**: Generate and download detailed analysis reports
- **Modern UI**: Clean, professional interface designed for medical applications

## Technologies Used

- **Deep Learning**: TensorFlow/Keras for neural network models
- **Web Framework**: Streamlit for the interactive interface
- **Visualization**: Plotly for interactive charts and OpenCV for image processing
- **AI Explanation**: Google's Generative AI (Gemini) for medical explanations
- **UI Components**: Custom components and animations

## Installation and Setup

1. Clone the repository
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Set up your Google API key for Gemini:
```
export GOOGLE_API_KEY=your_api_key_here
```
4. Run the application:
```
streamlit run app.py
```

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and model inference code
- `xception_model.weights.h5`: Transfer learning model based on Xception architecture
- `cnn_model.h5`: Custom CNN model for brain tumor classification
- `requirements.txt`: Python dependencies
- `saliency_maps/`: Directory for storing generated saliency maps
- `reports/`: Directory for storing generated analysis reports

## Model Details

### Xception Transfer Learning Model
- Base model: Xception pre-trained on ImageNet
- Additional layers: Dropout and Dense layers for classification
- Input size: 299×299×3
- Classes: 4 (Glioma, Meningioma, No Tumor, Pituitary)

### Custom CNN Model
- Architecture: Custom convolutional neural network
- Input size: 224×224×3
- Classes: 4 (Glioma, Meningioma, No Tumor, Pituitary)

## Disclaimer

NeuroLens is a demonstration tool and is not FDA-approved for clinical use. All predictions should be verified by qualified healthcare professionals. This application is intended for research and educational purposes only.

## License

[MIT License](LICENSE)
