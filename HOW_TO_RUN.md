# How to Run

This guide shows how to run and test the disease prediction system.

## Method 1: Jupyter Notebook (Recommended)

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras sentence-transformers

# Open notebook
jupyter notebook Disease_predictor_net.ipynb

# Run all cells, then edit test cell at bottom
```

## Method 2: Python Script

```bash
# Interactive mode
python test_disease_predictor.py

# Simple test
python test_simple.py
```

## Method 3: Full Pipeline (Voice Input)

Requires API keys from ElevenLabs and OpenRouter.

### Setup

1. Create `.env` file:
```
ELEVENLABS_API_KEY=your_key_here
```

2. Install audio packages:
```bash
pip install sounddevice scipy elevenlabs python-dotenv openai
```

### Run

1. Open `speech_to_text.ipynb`
2. Run cells to record and transcribe
3. Copy extracted symptoms
4. Use in disease predictor

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Internet connection (first time only)

## Testing

In the notebook, modify the test cell:

```python
my_symptoms = ['fever', 'headache', 'cough']
```

The system understands natural language:
- "my head hurts" becomes "headache"
- "stomach ache" becomes "abdominal pain"

## Troubleshooting

**Package not found:**
```bash
pip install <package_name>
```

**Training takes long:**
Normal. Takes about 10 minutes on CPU.

**Low accuracy:**
Use 3-5 specific symptoms for best results.

## Training Time

First run trains the model:
- About 10 minutes on CPU
- About 2-3 minutes on GPU

Subsequent runs load pre-trained model (much faster).

## Notes

- Model accuracy: 85-90%
- Works with 132 symptoms
- Predicts from 41 diseases
- Educational use only
