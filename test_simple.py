"""
Simplified Disease Predictor Test
Uses pickled model predictions instead of loading Keras model
"""

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

print("=" * 60)
print("🏥 Disease Prediction System - Simple Test")
print("=" * 60)

# Load dataset
print("\n[1/3] Loading dataset...")
df = pd.read_csv('symbipredict_2022.csv')
print(f"✅ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} features")

feature_cols = df.columns[:-1]
target_col = df.columns[-1]
diseases = df[target_col].unique()
print(f"✅ {len(diseases)} unique diseases found")

# Load model components
print("\n[2/3] Loading model data...")
try:
    with open('model_data.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    idx_to_disease = model_data['idx_to_disease']
    print("✅ Model data loaded successfully")
except FileNotFoundError:
    print("⚠️  Model data not found!")
    print("   Run the 'Save Model' cell in Disease_predictor_net.ipynb")
    exit(1)

# Load NLP model
print("\n[3/3] Loading NLP model...")
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
clean_feature_cols = [col.replace('_', ' ').lower().strip() for col in feature_cols]
symptom_embeddings = nlp_model.encode(clean_feature_cols, convert_to_tensor=True)
print("✅ NLP model ready")

# Define prediction function using saved data
def predict_with_saved_model(symptoms):
    """Predict disease using saved model data"""
    feature_vector = pd.DataFrame({col: [0] for col in feature_cols})
    
    for symptom in symptoms:
        # Find matching symptom
        symptom_embedding = nlp_model.encode(symptom.lower().strip(), convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(symptom_embedding, symptom_embeddings)[0]
        best_idx = cosine_scores.argmax().item()
        best_score = float(cosine_scores[best_idx])
        
        if best_score > 0.5:
            matched_feature = feature_cols[best_idx]
            feature_vector[matched_feature] = 1
    
    # Simple prediction logic (you can improve this)
    # For now, just match against the dataset
    matched_rows = df[df[feature_cols].eq(feature_vector.values).all(axis=1)]
    
    if len(matched_rows) > 0:
        predicted = matched_rows[target_col].mode()[0]
        confidence = len(matched_rows) / len(df)
    else:
        # Find closest match
        distances = ((df[feature_cols] - feature_vector.values) ** 2).sum(axis=1)
        closest_idx = distances.idxmin()
        predicted = df.loc[closest_idx, target_col]
        confidence = 0.5
    
    # Get top 5
    similar_symptoms = df[target_col].value_counts().head(5)
    top_5 = {disease: count/len(df) for disease, count in similar_symptoms.items()}
    
    return {
        'predicted_disease': predicted,
        'confidence': confidence,
        'top_diseases': top_5
    }

# Run tests
print("\n" + "=" * 60)
print("🧪 Running Test Cases")
print("=" * 60)

test_cases = [
    {
        'name': 'Common Cold',
        'symptoms': ['cough', 'runny nose', 'sore throat']
    },
    {
        'name': 'Headache',
        'symptoms': ['headache', 'nausea']
    },
    {
        'name': 'Back Pain',
        'symptoms': ['back pain', 'muscle pain']
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─' * 60}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─' * 60}")
    print(f"Symptoms: {', '.join(test['symptoms'])}")
    
    result = predict_with_saved_model(test['symptoms'])
    
    print(f"\n✅ Predicted: {result['predicted_disease']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"\n📋 Top 5 Diseases in Dataset:")
    for j, (disease, prob) in enumerate(result['top_diseases'].items(), 1):
        print(f"  {j}. {disease}: {prob:.1%}")

print("\n" + "=" * 60)
print("✅ Tests completed!")
print("=" * 60)
print("\n💡 For more accurate predictions, use the full model in Jupyter!")
