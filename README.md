# Disease Prediction System

A neural network that predicts diseases from symptoms. Trained on 4,961 medical cases with 132 symptoms and 41 diseases.

## Overview

This project demonstrates how machine learning can be applied to medical diagnosis. It takes symptom descriptions in natural language and returns disease predictions with confidence scores.

What makes it interesting:
- Combines multiple AI technologies: deep learning for classification, NLP for understanding symptoms, and speech recognition for voice input
- Handles natural language flexibly (understands "my head hurts" as "headache")
- Supports Arabic dialects with automatic translation
- Achieves 85-90% accuracy on medical data

Components:
- Disease prediction model (Jupyter notebook)
- Speech-to-text with Arabic support (optional)
- Python test scripts

## Key Technologies

- **TensorFlow/Keras**: Deep learning framework for building the neural network
- **Sentence Transformers**: NLP library for semantic similarity matching of symptoms
- **ElevenLabs API**: Speech-to-text transcription with dialect support
- **OpenAI GPT-3.5**: Language translation and symptom extraction
- **Scikit-learn**: Data preprocessing and train/test splitting

## What I Learned

Building this project taught me:
- How to work with medical datasets and handle imbalanced classes
- Semantic similarity matching using sentence embeddings
- Integrating multiple APIs (speech, translation, NLP) into one pipeline
- Challenges of serializing and loading trained models across different frameworks
- Importance of natural language processing in making AI accessible
- Building end-to-end ML systems from data exploration to deployment

## Quick Start

1. Install dependencies:
```bash
pip install pandas numpy tensorflow keras scikit-learn matplotlib seaborn sentence-transformers
```

2. Run the notebook:
```bash
jupyter notebook Disease_predictor_net.ipynb
```

3. Run all cells, then edit the test cell with your symptoms:
```python
my_symptoms = ['fever', 'headache', 'cough']
```

## Files

- `Disease_predictor_net.ipynb` - Main model
- `speech_to_text.ipynb` - Voice input
- `symbipredict_2022.csv` - Dataset
- `test_disease_predictor.py` - Interactive test
- `test_simple.py` - Basic test

## Using Python Scripts

```bash
python test_disease_predictor.py
```

Or for simple testing:
```bash
python test_simple.py
```

## Voice Input (Optional)

Requires API keys from ElevenLabs and OpenRouter.

1. Create `.env` file:
```
ELEVENLABS_API_KEY=your_key
```

2. Install audio packages:
```bash
pip install sounddevice scipy elevenlabs python-dotenv openai
```

3. Open `speech_to_text.ipynb` and follow the cells.

## Model Details

- Architecture: 512 → 256 → 128 → 41 neurons with ReLU activation
- Regularization: Dropout (0.3, 0.2) and Batch Normalization
- Optimizer: Adam with learning rate 0.0001
- Loss function: Sparse Categorical Crossentropy
- Validation accuracy: 85-90%
- Training time: ~10 minutes on CPU
- NLP: sentence-transformers (all-MiniLM-L6-v2) for semantic matching

## Dataset

- 4,961 cases
- 132 symptoms (binary features)
- 41 diseases

## Notes

This is an educational project built as part of an NLP course. It demonstrates the practical application of machine learning in healthcare, but is not intended for medical diagnosis. Always consult healthcare professionals for medical concerns.

The most challenging part was integrating different technologies and handling model serialization issues between TensorFlow and Keras formats.

## License

Educational use only.
