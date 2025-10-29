# 🚀 HOW TO RUN THIS PROJECT

This guide shows you exactly how to run and test the Disease Prediction System.

---

## 🎯 QUICKEST WAY (Recommended)

### Just Use Jupyter Notebook!

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras sentence-transformers

# 2. Open and run the notebook
jupyter notebook Disease_predictor_net.ipynb

# 3. Run ALL cells from top to bottom
# 4. At the end, change symptoms in the test cell and run it again!
```

**That's it!** No file saving, no loading, no compatibility issues! 🎉

---

## 📋 Prerequisites

- ✅ Python 3.8 or higher installed
- ✅ Jupyter Notebook or JupyterLab
- ✅ Internet connection (for downloading models first time)

---

## 🎤 Method 2: FULL PIPELINE (Voice → Prediction)

### What You'll Need:
- ElevenLabs API key (get free trial at https://elevenlabs.io)
- OpenRouter API key (for GPT-3.5, get at https://openrouter.ai)

### Steps:

1. **Setup API Keys**
   ```bash
   # Create .env file in project folder
   echo ELEVENLABS_API_KEY=your_key_here > .env
   ```

2. **Record & Transcribe Speech**
   ```bash
   jupyter notebook speech_to_text.ipynb
   ```
   
   Execute these cells:
   - **Cell 1-2**: Records 10 seconds of audio (speak your symptoms)
   - **Cell 3-6**: Transcribes using ElevenLabs (Arabic supported)
   - **Cell 7-8**: Translates to English using GPT-3.5
   - **Cell 9**: Extracts symptom list
   
   Copy the extracted symptoms (e.g., `['headache', 'fever']`)

3. **Predict Disease**
   ```bash
   jupyter notebook Disease_predictor_net.ipynb
   ```
   
   - Run all cells to train (if not done already)
   - Go to the prediction section at the bottom
   - Paste your symptoms:
     ```python
     my_symptoms = ['headache', 'fever', 'cough']  # Your symptoms
     result = predict_disease_from_symptoms(my_symptoms, 'symbi')
     print(result)
     ```

---

## 💡 Method 3: DIRECT TEXT INPUT (No Voice)

### Quick Test in Jupyter:

```python
# In Disease_predictor_net.ipynb, after training:

# Test 1: Simple symptoms
test_symptoms = ['headache', 'fever', 'cough']
result = predict_disease_from_symptoms(test_symptoms, 'symbi')
print(f"Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.1%}")

# Test 2: Natural language (the system is smart!)
symptoms = ['my back hurts', 'feeling tired', 'body aches']
result = predict_disease_from_symptoms(symptoms, 'symbi')
# It automatically maps to proper medical terms

# Test 3: See top 5 predictions
for disease, prob in result['top_diseases'].items():
    print(f"{disease}: {prob:.1%}")
```

---

## 🧪 Sample Test Cases

Try these in the notebook or test script:

```python
# Common Cold
predict_disease_from_symptoms(
    ['cough', 'runny nose', 'sore throat', 'fatigue'], 
    'symbi'
)

# Migraine
predict_disease_from_symptoms(
    ['headache', 'nausea', 'sensitivity to light'], 
    'symbi'
)

# Flu
predict_disease_from_symptoms(
    ['fever', 'chills', 'body aches', 'fatigue'], 
    'symbi'
)

# Gastro Issue
predict_disease_from_symptoms(
    ['abdominal pain', 'nausea', 'vomiting', 'diarrhea'], 
    'symbi'
)
```

---

## 📊 What You'll See

### Training Output:
```
Epoch 1/30
117/117 [==============================] - 2s 14ms/step
- loss: 2.3456 - accuracy: 0.4523 - val_loss: 1.9876 - val_accuracy: 0.5234

...

Epoch 15/30
117/117 [==============================] - 1s 12ms/step
- loss: 0.4321 - accuracy: 0.8876 - val_loss: 0.5432 - val_accuracy: 0.8523

✅ Model trained successfully!
```

### Prediction Output:
```json
{
    "predicted_disease": "Common Cold",
    "confidence": 0.8523,
    "top_diseases": {
        "Common Cold": 0.8523,
        "Flu": 0.0892,
        "Sinusitis": 0.0234,
        "Bronchitis": 0.0156,
        "Allergic Rhinitis": 0.0089
    },
    "cleaned_symptoms": ["cough", "runny nose", "sore throat"]
}
```

---

## ⚠️ Troubleshooting

### "No module named 'tensorflow'"
```bash
pip install tensorflow keras
```

### "API key not found"
```bash
# Make sure .env file exists and contains:
ELEVENLABS_API_KEY=your_actual_key_here
```

### "Model file not found"
```python
# In Disease_predictor_net.ipynb, run this at the end:
symbi_model.save('disease_model.h5')
```

### Training takes too long
```python
# Reduce epochs in the training cell:
epochs=10  # Instead of 30
```

### Audio recording not working
```bash
# Windows:
pip install sounddevice scipy

# If still not working:
pip install pyaudio
```

---

## 📈 Expected Performance

- **Training Time**: 5-15 minutes on CPU
- **Prediction Time**: < 1 second per query
- **Accuracy**: ~85-90% on validation set
- **Memory Usage**: ~500MB during training

---

## 🎓 Learning Path

1. **Day 1**: Install, train model, test with sample symptoms
2. **Day 2**: Explore data visualizations, understand neural network
3. **Day 3**: Test voice input pipeline
4. **Day 4**: Experiment with different symptoms and analyze results
5. **Day 5**: Modify model architecture, try improvements

---

## 💬 Need Help?

1. Check the main README.md for detailed documentation
2. Look at the notebook markdown cells for explanations
3. Run `python test_disease_predictor.py` for automated tests
4. Open an issue on GitHub if you find bugs

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`setup.bat` or `pip install ...`)
- [ ] Jupyter Notebook/Lab working
- [ ] Can open and run Disease_predictor_net.ipynb
- [ ] Model trained successfully (see accuracy metrics)
- [ ] Can make predictions with test symptoms
- [ ] (Optional) API keys configured for voice input
- [ ] (Optional) Can record and transcribe audio

---

**You're ready to go! Start with Method 1 for the quickest results.** 🚀
