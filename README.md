# 🏥 Disease Prediction System with Voice Input

An AI-powered medical diagnosis system that combines **speech recognition**, **natural language processing (NLP)**, and **deep neural networks** to predict diseases based on symptoms described in voice or text format.

> **📖 New to this project?** Check out [HOW_TO_RUN.md](HOW_TO_RUN.md) for step-by-step instructions!

## 📋 Project Overview

This project consists of two main components:

1. **Speech-to-Text Processing** (`speech_to_text.ipynb`): Records audio, transcribes it (including Arabic dialect support), translates to English, and extracts symptoms using GPT-3.5.

2. **Disease Prediction Model** (`Disease_predictor_net.ipynb`): A deep learning neural network trained on symptom-disease associations that predicts diseases with confidence scores.

### Key Features

- 🎤 **Voice Recording & Transcription**: Records audio and converts it to text using ElevenLabs API
- 🌍 **Multilingual Support**: Transcribes Arabic dialects and translates to English
- 🧠 **Deep Neural Network**: Trained on 132 symptoms to predict from 41 diseases
- 🔍 **Intelligent Symptom Matching**: Uses sentence transformers to match user descriptions to medical terminology
- 📊 **Confidence Scores**: Provides top 5 disease predictions with probability scores
- 📈 **Visualization**: Comprehensive data analysis and model performance metrics

---

## 🏗️ Architecture

### Component 1: Speech-to-Text Pipeline

```
Audio Recording → ElevenLabs Transcription → OpenRouter GPT-3.5 Translation → Symptom Extraction → Disease Prediction
```

**Process Flow:**
1. Records 10 seconds of audio (configurable)
2. Transcribes audio using ElevenLabs Scribe API (supports Arabic dialects)
3. Translates to English using GPT-3.5 via OpenRouter
4. Extracts structured symptom list from conversational text
5. Passes symptoms to disease prediction model

### Component 2: Neural Network Disease Predictor

**Model Architecture:**
```
Input Layer (n symptoms)
    ↓
Dense (512 neurons) + BatchNorm + ReLU
    ↓
Dense (256 neurons) + Dropout (0.3) + ReLU
    ↓
Dense (128 neurons) + Dropout (0.2) + ReLU
    ↓
Output Layer (n diseases) + Softmax
```

**Features:**
- **Dataset**: `symbipredict_2022.csv` with one-hot encoded symptoms
- **Training**: 70/30 train-validation split with early stopping
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Sparse Categorical Crossentropy
- **NLP Matching**: Uses `all-MiniLM-L6-v2` sentence transformer for symptom similarity

---

## 📁 Project Structure

```
MiniProjet_NLP/
│
├── Disease_predictor_net.ipynb     # Neural network model & training
├── speech_to_text.ipynb            # Audio recording & transcription
├── symbipredict_2022.csv           # Training dataset
├── test_disease_predictor.py       # Quick test script
├── setup.bat                       # Automated setup (Windows)
├── run_test.bat                    # Quick test runner (Windows)
├── output.wav                      # Recorded audio file
├── test2.mp3                       # Sample audio file
├── Record (online-voice-recorder.com).mp3  # Sample recording
└── README.md                       # This file
```

---

## ⚡ Quick Start Guide

### 🎯 Fastest Way to Get Started (Windows)

```bash
# 1. Install dependencies
setup.bat

# 2. Train the model (open in Jupyter)
jupyter notebook Disease_predictor_net.ipynb
# Run all cells, then add: symbi_model.save('disease_model.h5')

# 3. Run quick test
run_test.bat
# Or: python test_disease_predictor.py
```

### Step-by-Step: Running the Complete Pipeline

#### **Method 1: Voice Input → Disease Prediction (Full Pipeline)**

1. **Setup Environment** (First time only)
   ```bash
   # Install all dependencies
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
   pip install sounddevice scipy sentence-transformers elevenlabs python-dotenv openai
   ```

2. **Configure API Keys**
   - Create `.env` file with your ElevenLabs API key
   - Add your OpenRouter API key in `speech_to_text.ipynb` (cells with OpenAI client)

3. **Run Speech-to-Text Pipeline**
   - Open `speech_to_text.ipynb` in Jupyter Notebook/Lab
   - Execute cells in order:
     ```
     Cell 1-2: Record audio (speak your symptoms for 10 seconds)
     Cell 3-6: Transcribe audio using ElevenLabs
     Cell 7-8: Translate from Arabic to English
     Cell 9: Extract symptoms from text
     ```
   - Copy the extracted symptoms array (e.g., `['headache', 'fever', 'shoulder pain']`)

4. **Run Disease Prediction**
   - Open `Disease_predictor_net.ipynb`
   - Execute all cells up to the training section (to load model)
   - Jump to the prediction cells at the bottom
   - Use your symptoms:
     ```python
     test_symptoms = ['headache', 'fever', 'shoulder pain']  # Your symptoms
     result = predict_disease_from_symptoms(test_symptoms, 'symbi')
     print(result)
     ```

#### **Method 2: Text Input Only (Quick Test)**

1. **Setup** (First time only)
   ```bash
   pip install pandas numpy tensorflow keras scikit-learn sentence-transformers matplotlib seaborn
   ```

2. **Train & Test the Model**
   - Open `Disease_predictor_net.ipynb`
   - Run all cells from top to bottom (takes ~10-15 minutes first time)
   - This will:
     - Load the dataset
     - Visualize data distribution
     - Train the neural network
     - Create prediction functions

3. **Test with Sample Symptoms**
   - Scroll to the bottom of the notebook
   - Find or create a cell with:
     ```python
     # Test with your own symptoms
     my_symptoms = ['back pain', 'fatigue', 'headache']
     validated_symptoms, _ = map_symptoms(my_symptoms, threshold=0.5)
     result = predict_disease_from_symptoms(validated_symptoms, 'symbi')
     
     print(f"Predicted Disease: {result['predicted_disease']}")
     print(f"Confidence: {result['confidence']:.2%}")
     print("\nTop 5 Predictions:")
     for disease, prob in result['top_diseases'].items():
         print(f"  - {disease}: {prob:.2%}")
     ```

#### **Method 3: Pre-trained Model Testing**

If you want to test without retraining:

```python
# In Disease_predictor_net.ipynb, after training once
# You can save the model
symbi_model.save('disease_model.h5')

# Later, load it directly
from tensorflow import keras
symbi_model = keras.models.load_model('disease_model.h5')

# Then test immediately
test_symptoms = ['cough', 'fever', 'fatigue']
result = predict_disease_from_symptoms(test_symptoms, 'symbi')
print(result)
```

---

## 🧪 Example Test Cases

### Test Case 1: Common Cold Symptoms
```python
symptoms = ['cough', 'runny nose', 'sore throat', 'fatigue']
result = predict_disease_from_symptoms(symptoms, 'symbi')
# Expected: Common Cold, Flu, or Upper Respiratory Infection
```

### Test Case 2: Headache-Related
```python
symptoms = ['headache', 'nausea', 'sensitivity to light']
result = predict_disease_from_symptoms(symptoms, 'symbi')
# Expected: Migraine, Tension Headache, or similar
```

### Test Case 3: Gastrointestinal
```python
symptoms = ['abdominal pain', 'nausea', 'vomiting', 'diarrhea']
result = predict_disease_from_symptoms(symptoms, 'symbi')
# Expected: Gastroenteritis, Food Poisoning, or similar
```

### Test Case 4: Using Natural Language
The system handles conversational descriptions:
```python
symptoms = ['my head hurts', 'feeling tired', 'body aches']
# System automatically maps to: ['headache', 'fatigue', 'muscle pain']
result = predict_disease_from_symptoms(symptoms, 'symbi')
```

---

## 🔍 Troubleshooting

### Common Issues

**Issue 1: "Module not found" error**
```bash
# Solution: Install missing package
pip install <package_name>
```

**Issue 2: API key errors**
```bash
# Solution: Check .env file exists and contains valid key
# Verify: print(os.getenv("ELEVENLABS_API_KEY"))
```

**Issue 3: Audio recording fails**
```bash
# Solution: Install audio backend
pip install sounddevice scipy
# On Windows, may need: pip install pyaudio
```

**Issue 4: Model training too slow**
```python
# Solution: Reduce epochs or use smaller batch size
epochs=10  # Instead of 30
batch_size=64  # Instead of 32
```

**Issue 5: Low prediction confidence**
```python
# Try:
# 1. Be more specific with symptoms
# 2. Include more symptoms (3-5 symptoms work best)
# 3. Use medical terminology when possible
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Microphone (for voice recording)

### Required Libraries

```bash
# Core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Deep learning
pip install tensorflow keras

# Audio processing
pip install sounddevice scipy

# NLP & Speech-to-Text
pip install openai-whisper sentence-transformers
pip install elevenlabs
pip install python-dotenv

# OpenRouter API client
pip install openai
```

### API Keys Setup

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

**Important**: You'll also need an OpenRouter API key for GPT-3.5 access. Currently, the API key is hardcoded in `speech_to_text.ipynb`. For better security:
- Add `OPENROUTER_API_KEY=your_key_here` to your `.env` file
- Update the notebook to load it: `api_key=os.getenv("OPENROUTER_API_KEY")`

---

## 💻 Usage

### Option 1: Voice Input → Disease Prediction

1. **Open `speech_to_text.ipynb`**
2. Run the audio recording cell:
   ```python
   # Records 10 seconds of audio
   # Speak your symptoms clearly
   ```
3. Run transcription and translation cells
4. Run symptom extraction cell
5. Copy extracted symptoms to `Disease_predictor_net.ipynb`
6. Run prediction cell to get disease predictions

### Option 2: Direct Text Input

1. **Open `Disease_predictor_net.ipynb`**
2. Use the prediction function directly:
   ```python
   test_symptoms = ['headache', 'fever', 'cough', 'fatigue']
   result = predict_disease_from_symptoms(test_symptoms, 'symbi')
   print(result)
   ```

### Example Output

```python
{
    'predicted_disease': 'Common Cold',
    'confidence': 0.8523,
    'top_diseases': {
        'Common Cold': 0.8523,
        'Flu': 0.0892,
        'Sinusitis': 0.0234,
        'Bronchitis': 0.0156,
        'Allergic Rhinitis': 0.0089
    },
    'cleaned_symptoms': ['headache', 'fever', 'cough', 'fatigue']
}
```

---

## 📊 Dataset

**Source**: `symbipredict_2022.csv`

**Structure**:
- **Rows**: 4,961 patient cases
- **Columns**: 133 total (132 binary symptom features + 1 disease label)
- **Format**: One-hot encoded (1 = symptom present, 0 = absent)
- **Diseases**: 41 unique disease categories

**Top Symptoms in Dataset**:
- Fever
- Headache
- Cough
- Fatigue
- Nausea
- And 127+ more...

---

## 🧪 Model Performance

### Training Results

- **Validation Accuracy**: ~85-90% (depends on training run)
- **Training Time**: ~5-10 minutes on CPU
- **Early Stopping**: Monitors validation accuracy with patience=10

### Visualizations Included

1. **Training/Validation Accuracy & Loss Curves**
2. **Disease Distribution Analysis**
3. **Top 20 Most Common Symptoms**
4. **Symptom Similarity Scores**
5. **Confusion Matrix** (can be added)

---

## 🔬 Technical Details

### Symptom Matching Algorithm

The system uses **semantic similarity** to match user descriptions to medical terms:

```python
# Example: "my head hurts" → "headache"
# Similarity score: 0.78

model = SentenceTransformer('all-MiniLM-L6-v2')
threshold = 0.5  # Minimum similarity to consider a match
```

**Benefits**:
- Handles typos and variations
- Understands synonyms (e.g., "stomach ache" → "abdominal pain")
- Works with conversational language

### Neural Network Details

- **Input**: Binary vector of symptom presence (size: 132)
- **Output**: Probability distribution over diseases (size: 41)
- **Activation**: ReLU (hidden), Softmax (output)
- **Regularization**: Dropout (0.2-0.3) + BatchNormalization
- **Training Strategy**: Early stopping to prevent overfitting

---

## ⚠️ Important Notes

### Medical Disclaimer

**This system is for educational and research purposes only.**

- ❌ NOT a substitute for professional medical advice
- ❌ NOT for self-diagnosis or treatment decisions
- ✅ Demonstrates AI/ML applications in healthcare
- ✅ Can assist healthcare professionals as a preliminary screening tool

**Always consult qualified healthcare providers for medical concerns.**

### Limitations

1. **Dataset Size**: Limited to conditions in training data
2. **Accuracy**: Model predictions depend on symptom description quality
3. **Language**: English symptoms work best; translation may introduce errors
4. **Context**: Cannot consider patient history, lab results, or physical examination

---

## 🛠️ Future Improvements

- [ ] Expand dataset with more diseases and symptoms
- [ ] Add multilingual support (direct Arabic symptom processing)
- [ ] Implement attention mechanisms for better feature importance
- [ ] Create web interface (Flask/Streamlit)
- [ ] Add patient history tracking
- [ ] Integrate with medical databases (ICD-10 codes)
- [ ] Implement explainable AI (LIME/SHAP) for transparency
- [ ] Add symptom duration and severity features

---

## 📚 References

### Libraries & Frameworks
- **TensorFlow/Keras**: Deep learning framework
- **Sentence Transformers**: NLP embeddings
- **ElevenLabs**: Speech-to-text API
- **OpenRouter**: GPT-3.5 API access
- **SoundDevice**: Audio recording

### Research Papers
- Neural networks for disease prediction
- Symptom-based diagnosis systems
- Healthcare NLP applications

---

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙋 Support

For questions or issues:
- Open an issue on GitHub
- Contact: [BaidarSamir](https://github.com/BaidarSamir)

---

## 🎓 Academic Context

This project was developed as part of an NLP Mini-Project to demonstrate:
- Integration of speech recognition with machine learning
- Real-world application of deep neural networks
- Healthcare AI system design
- End-to-end ML pipeline development

---

**Built with ❤️ for advancing AI in Healthcare**
