"""
Quick Test Script for Disease Predictor
Run this after training the model in Disease_predictor_net.ipynb
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from sentence_transformers import SentenceTransformer, util

# Configuration
CSV_FILE = 'symbipredict_2022.csv'
MODEL_FILE = 'disease_model.h5'  # Optional: save/load model

def test_prediction():
    """Test disease prediction with sample symptoms"""
    
    print("=" * 60)
    print("🏥 Disease Prediction System - Quick Test")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Dataset loaded: {df.shape[0]} cases, {df.shape[1]} features")
    except FileNotFoundError:
        print(f"❌ Error: {CSV_FILE} not found!")
        return
    
    # Prepare feature columns
    feature_cols = df.columns[:-1]  # All columns except last (disease)
    target_col = df.columns[-1]
    
    # Get disease mappings
    diseases = df[target_col].unique()
    disease_to_idx = {disease: idx for idx, disease in enumerate(diseases)}
    idx_to_disease = {idx: disease for disease, idx in disease_to_idx.items()}
    
    print(f"✅ {len(diseases)} unique diseases found")
    
    # Load or check for model
    print("\n[2/4] Checking for trained model...")
    try:
        model = keras.models.load_model(MODEL_FILE)
        print(f"✅ Model loaded from {MODEL_FILE}")
    except:
        print(f"⚠️  No saved model found. Please train model first in Disease_predictor_net.ipynb")
        print("   Then save it with: symbi_model.save('disease_model.h5')")
        return
    
    # Load sentence transformer for symptom matching
    print("\n[3/4] Loading NLP model for symptom matching...")
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    clean_feature_cols = [col.replace('_', ' ').lower().strip() for col in feature_cols]
    symptom_embeddings = nlp_model.encode(clean_feature_cols, convert_to_tensor=True)
    print("✅ NLP model ready")
    
    # Test cases
    print("\n[4/4] Running test cases...")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Test Case 1: Common Cold',
            'symptoms': ['cough', 'runny nose', 'sore throat', 'fatigue']
        },
        {
            'name': 'Test Case 2: Headache',
            'symptoms': ['headache', 'nausea', 'sensitivity to light']
        },
        {
            'name': 'Test Case 3: Back Pain',
            'symptoms': ['back pain', 'muscle pain', 'stiffness']
        },
        {
            'name': 'Test Case 4: Fever',
            'symptoms': ['fever', 'chills', 'body aches', 'fatigue']
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"🧪 {test_case['name']}")
        print(f"{'─' * 60}")
        print(f"Input Symptoms: {', '.join(test_case['symptoms'])}")
        
        # Create feature vector
        feature_vector = pd.DataFrame({col: [0] for col in feature_cols})
        
        # Map symptoms to features
        matched_symptoms = []
        for symptom in test_case['symptoms']:
            symptom_embedding = nlp_model.encode(symptom.lower().strip(), convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(symptom_embedding, symptom_embeddings)[0]
            
            # Get best match
            best_idx = cosine_scores.argmax().item()
            best_score = float(cosine_scores[best_idx])
            
            if best_score > 0.5:
                matched_feature = feature_cols[best_idx]
                feature_vector[matched_feature] = 1
                matched_symptoms.append(f"{matched_feature} ({best_score:.2f})")
        
        print(f"Matched Features: {', '.join(matched_symptoms)}")
        
        # Make prediction
        predictions = model.predict(feature_vector.values, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_indices = predictions.argsort()[-3:][::-1]
        
        print(f"\n🎯 Predictions:")
        for rank, idx in enumerate(top_3_indices, 1):
            disease = idx_to_disease[idx]
            confidence = predictions[idx]
            bar = '█' * int(confidence * 20)
            print(f"  {rank}. {disease}")
            print(f"     Confidence: {confidence:.1%} {bar}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    
    # Interactive mode
    print("\n💡 Want to test your own symptoms? (y/n): ", end='')
    try:
        response = input().lower()
        if response == 'y':
            print("\nEnter your symptoms (comma-separated):")
            print("Example: headache, fever, cough, fatigue")
            user_input = input("➤ ")
            
            user_symptoms = [s.strip() for s in user_input.split(',')]
            
            # Create feature vector
            feature_vector = pd.DataFrame({col: [0] for col in feature_cols})
            
            matched_symptoms = []
            for symptom in user_symptoms:
                symptom_embedding = nlp_model.encode(symptom.lower().strip(), convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(symptom_embedding, symptom_embeddings)[0]
                
                best_idx = cosine_scores.argmax().item()
                best_score = float(cosine_scores[best_idx])
                
                if best_score > 0.5:
                    matched_feature = feature_cols[best_idx]
                    feature_vector[matched_feature] = 1
                    matched_symptoms.append(f"{matched_feature.replace('_', ' ')} ({best_score:.2f})")
            
            print(f"\n🔍 Matched symptoms: {', '.join(matched_symptoms)}")
            
            # Make prediction
            predictions = model.predict(feature_vector.values, verbose=0)[0]
            top_5_indices = predictions.argsort()[-5:][::-1]
            
            print(f"\n🎯 Top 5 Predicted Diseases:")
            for rank, idx in enumerate(top_5_indices, 1):
                disease = idx_to_disease[idx]
                confidence = predictions[idx]
                bar = '█' * int(confidence * 30)
                print(f"  {rank}. {disease}")
                print(f"     {confidence:.1%} {bar}")
            
            print("\n⚠️  Reminder: This is for educational purposes only!")
            print("   Always consult a healthcare professional for medical advice.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during interactive test: {e}")

if __name__ == "__main__":
    try:
        test_prediction()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model in Disease_predictor_net.ipynb")
        print("2. Saved it with: symbi_model.save('disease_model.h5')")
        print("3. Installed dependencies: pip install tensorflow keras sentence-transformers pandas")
