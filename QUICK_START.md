# Quick Start Guide

## Easiest Way to Use

This is the simplest method - everything runs in one notebook.

```bash
# 1. Install packages
pip install pandas numpy tensorflow keras scikit-learn matplotlib seaborn sentence-transformers

# 2. Open notebook
jupyter notebook Disease_predictor_net.ipynb

# 3. Run all cells

# 4. Edit symptoms in test cell and run again
```

## File Structure

```
MiniProjet_NLP/
├── Disease_predictor_net.ipynb    # Main file
├── speech_to_text.ipynb           # Voice input (optional)
├── symbipredict_2022.csv          # Dataset
├── test_disease_predictor.py      # Interactive test
├── test_simple.py                 # Simple test
└── setup.bat                      # Windows installer
```

## Testing

In Jupyter, find the test cell at the bottom:

```python
# Change these symptoms
my_symptoms = ['fever', 'headache', 'cough', 'fatigue']
```

Run the cell to get predictions.

Examples to try:
- `['back pain', 'stiffness']`
- `['stomach pain', 'vomiting', 'nausea']`
- `['cough', 'fever', 'difficulty breathing']`

## Using Python Scripts

```bash
python test_disease_predictor.py
```

This includes an interactive mode where you can type your own symptoms.

## Notes

- Training takes about 10 minutes
- Works best with 2-5 symptoms
- System understands natural language
- Educational use only - not for medical diagnosis
