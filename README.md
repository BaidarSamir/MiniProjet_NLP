# 🏥 AI Disease Predictor

Hey! This is a smart disease prediction system that listens to your symptoms and predicts what you might have. You can either speak your symptoms out loud (it even understands Arabic!) or just type them in. It uses deep learning and NLP to figure out what's going on.

> **� Want to jump right in?** Check out [QUICK_START.md](QUICK_START.md) for the simplest guide!

## 🤔 What Does This Thing Do?

Imagine you're not feeling well. Instead of googling your symptoms and convincing yourself you have 10 different diseases, you can use this tool to get a smart prediction based on actual medical data!

**Here's what happens:**

1. 🎤 **Tell it how you feel** - Speak in English or Arabic, or just type your symptoms
2. 🧠 **AI understands you** - It figures out what symptoms you're describing (even if you say "my head hurts" instead of "headache")  
3. 🔬 **Gets predictions** - A neural network trained on thousands of cases gives you the top possible diseases
4. 📊 **Shows confidence** - You get percentages so you know how sure the AI is

**Two main parts:**

- **Speech-to-Text** (`speech_to_text.ipynb`): Record your voice, it transcribes and translates if needed
- **Disease Predictor** (`Disease_predictor_net.ipynb`): The actual AI that predicts diseases from symptoms

### Cool Features

- � Works with voice input
- 🌍 Understands Arabic and translates automatically  
- 🧠 Trained on 132 symptoms and 41 diseases (almost 5,000 cases!)
- 🔍 Smart enough to understand "stomach ache" and "abdominal pain" mean the same thing
- 📊 Gives you top 5 predictions with confidence scores
- 📈 Shows you cool visualizations of the data

## 🏗️ How It Works (The Nerdy Stuff)

### Voice Pipeline (Optional - if you want voice input)

```
You speak → Records audio → Transcribes it → Translates to English → Extracts symptoms → Predicts disease
```

We use:
- **ElevenLabs** for transcription (it's really good with dialects!)
- **GPT-3.5** to translate and extract clean symptom lists from your rambling 😄

### The AI Brain

It's a neural network that looks like this:

```
Your symptoms (as numbers)
    ↓
512 neurons (lots of thinking!)
    ↓
256 neurons (getting focused)
    ↓
128 neurons (almost there...)
    ↓
41 possible diseases (the answer!)
```

**The secret sauce:**
- Trained on almost 5,000 patient cases
- Uses dropout and batch normalization (fancy terms that make it smarter)
- Gets about 85-90% accuracy on validation data
- Uses something called "sentence transformers" to understand that "belly pain" = "stomach ache"

## 📁 What's in Here?

```
MiniProjet_NLP/
│
├── Disease_predictor_net.ipynb     # The main AI model (start here!)
├── speech_to_text.ipynb            # Voice recording stuff (optional)
├── symbipredict_2022.csv           # The training data (4,961 cases)
├── test_disease_predictor.py       # Quick Python script to test
├── test_simple.py                  # Even simpler test script
├── setup.bat                       # Windows installer (double-click to setup)
├── QUICK_START.md                  # Easiest way to get started
├── HOW_TO_RUN.md                   # Detailed instructions
└── README.md                       # You're reading it! 👋
```

## ⚡ How to Use It

### The Super Easy Way (Windows)

```bash
# Step 1: Install everything you need
setup.bat

# Step 2: Open the notebook
jupyter notebook Disease_predictor_net.ipynb

# Step 3: Click "Run All" and wait about 10 minutes

# Step 4: At the end, change the symptoms and run that cell again!
```

That's literally it! 🎉

### The "I Want to Type Symptoms" Way

1. **First time setup:**
   ```bash
   pip install pandas numpy tensorflow keras scikit-learn matplotlib seaborn sentence-transformers
   ```

2. **Run the notebook:**
   ```bash
   jupyter notebook Disease_predictor_net.ipynb
   ```

3. **Run all cells from top to bottom** (Kernel → Restart & Run All)

4. **At the bottom, you'll see something like this:**
   ```python
   # Change these to your symptoms!
   my_symptoms = ['fever', 'headache', 'cough', 'fatigue']
   ```

5. **Change the symptoms to whatever you want** and run that cell again

**Try these examples:**
- `['back pain', 'stiffness', 'weakness']`
- `['stomach pain', 'vomiting', 'diarrhea']`
- `['cough', 'fever', 'difficulty breathing']`
- `['headache', 'dizziness', 'nausea']`
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


### Want to Use Voice Input? (Optional but Cool!)

This is a bit more advanced but totally doable:

1. **Get API keys** (you'll need these):
   - [ElevenLabs API](https://elevenlabs.io) for transcription
   - [OpenRouter API](https://openrouter.ai) for GPT-3.5

2. **Install extra stuff:**
   ```bash
   pip install sounddevice scipy elevenlabs python-dotenv openai
   ```

3. **Set up your keys:**
   - Create a file called `.env` 
   - Add: `ELEVENLABS_API_KEY=your_key_here`
   - In `speech_to_text.ipynb`, add your OpenRouter key

4. **Open `speech_to_text.ipynb`** and follow along:
   - Record audio (it'll record for 10 seconds)
   - It transcribes what you said
   - Translates if you spoke Arabic
   - Extracts symptoms
   - Copy those symptoms to the disease predictor!

---

## 🧪 Try These Examples

Once you have the notebook running, try these in the test cell:

```python
# Flu-like symptoms
my_symptoms = ['fever', 'cough', 'fatigue', 'body aches']

# Headache stuff  
my_symptoms = ['headache', 'nausea', 'sensitivity to light']

# Stomach problems
my_symptoms = ['abdominal pain', 'vomiting', 'diarrhea']

# Back pain
my_symptoms = ['back pain', 'stiffness', 'weakness']

# Heart-related (scary!)
my_symptoms = ['chest pain', 'shortness of breath', 'sweating']
```

The AI is smart enough to understand casual language too:
- "my head hurts" → understands as "headache"
- "stomach ache" → understands as "abdominal pain"  
- "can't breathe well" → understands as "difficulty breathing"

---

## � Troubleshooting (When Stuff Breaks)

**"Module not found"**
```bash
pip install whatever_is_missing
```

**"Can't record audio"**
```bash
pip install sounddevice scipy
# Windows might need: pip install pyaudio
```

**"Training takes forever"**  
Yeah, it takes about 10 minutes on a normal laptop. Grab a coffee! ☕

**"Predictions seem wrong"**  
- Use 3-5 symptoms (not just one)
- Be specific: "migraine" is better than "hurts"
- Try medical terms when you know them

**"API errors with voice input"**  
Check your `.env` file has the right API key and it's not expired

---

## 📊 The Dataset

Using `symbipredict_2022.csv`:
- **4,961 patient cases** from real medical records
- **132 different symptoms** (like fever, cough, pain, etc.)
- **41 possible diseases** it can predict
- Each case is a real diagnosis with recorded symptoms

The data is formatted as 1s and 0s (symptom present or not), which is perfect for machine learning!

---

## 🧠 Technical Details (For the Curious)

**Model specs:**
- Architecture: Sequential neural network
- Layers: 512 → 256 → 128 → 41 neurons
- Activation: ReLU (hidden layers), Softmax (output)
- Regularization: Dropout (0.3 and 0.2) + Batch Normalization
- Optimizer: Adam with learning rate 0.0001
- Loss: Sparse Categorical Crossentropy
- Training: 70/30 split, early stopping (patience=10)
- **Accuracy: ~85-90% on validation set** 🎯

**NLP Component:**
- Uses `sentence-transformers` library
- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Cosine similarity matching for symptoms
- Threshold: 0.5 for matching confidence

---

## ⚠️ Important Disclaimer

**THIS IS A STUDENT PROJECT FOR EDUCATIONAL PURPOSES ONLY!**

- 🚫 NOT a substitute for real medical advice
- 🚫 NOT approved by any medical authority  
- 🚫 DON'T use this to diagnose yourself or others
- ✅ DO use it to learn about AI and machine learning
- ✅ DO see a real doctor if you're actually sick!

Think of this as a fun demo of what AI can do, not as a real medical tool. If you're not feeling well, go see a doctor! 👨‍⚕️👩‍⚕️

---

## 🤝 Contributing

Got ideas? Found bugs? Want to add more diseases or symptoms? Feel free to:
- Open an issue
- Submit a pull request
- Fork it and make it your own!

---

## 📝 License

This is a student project. Use it for learning, modify it, share it, just remember the disclaimer above!

---

## � Credits

Built using:
- TensorFlow/Keras for the neural network
- Sentence Transformers for NLP
- ElevenLabs for transcription
- OpenAI GPT-3.5 for translation
- A lot of coffee and debugging ☕🐛

---

**Made with ❤️ for learning AI and helping understand how machine learning works in healthcare!**

*Remember: Real doctors went to school for like 10+ years. This AI trained for 10 minutes. Know the difference!* 😄
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
---

## 🎓 Academic Note

This was built as an NLP mini-project to learn about:
- How AI can work in healthcare
- Combining speech recognition with machine learning
- Building end-to-end AI pipelines
- The challenges (and fun!) of medical AI

---

## � Questions?

Something not working? Want to contribute? 
- Open an issue on GitHub
- Fork it and make improvements!
- Contact: [BaidarSamir](https://github.com/BaidarSamir)

---

**Built with ❤️ for learning how AI can help in healthcare**

*Just remember: This is a learning project. For real health problems, see real doctors!* 🏥
