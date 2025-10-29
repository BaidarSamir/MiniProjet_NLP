"""
Disease Predictor Test Script
Run this after training the model in Disease_predictor_net.ipynb
"""

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

# Configuration
CSV_FILE = 'symbipredict_2022.csv'
MODEL_DATA_FILE = 'model_data.pkl'

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
    
    # Load model data
    print("\n[2/4] Loading model data...")
    try:
        with open(MODEL_DATA_FILE, 'rb') as f:
            model_data = pickle.load(f)
        idx_to_disease = model_data['idx_to_disease']
        print(f"✅ Model data loaded from {MODEL_DATA_FILE}")
    except FileNotFoundError:
        print(f"⚠️  Error: {MODEL_DATA_FILE} not found!")
        print(f"   Please train the model first in Disease_predictor_net.ipynb")
        print("   Then run the 'Save Model Data' cell at the end")
        return
    except Exception as e:
        print(f"❌ Error loading model data: {e}")
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
        
        # Make prediction using dataset matching
        feature_vector_values = feature_vector.values
        
        # Find exact matches first
        matched_rows = df[df[feature_cols].eq(feature_vector_values).all(axis=1)]
        
        if len(matched_rows) > 0:
            # Found exact matches
            predicted_disease = matched_rows[target_col].mode()[0]
            confidence = len(matched_rows) / len(df)
            
            # Get distribution of diseases in matched rows
            disease_counts = matched_rows[target_col].value_counts()
            top_diseases = {}
            for disease, count in disease_counts.head(3).items():
                disease_idx = disease_to_idx[disease]
                top_diseases[disease_idx] = count / len(matched_rows)
        else:
            # Find closest match using distance
            distances = ((df[feature_cols] - feature_vector_values) ** 2).sum(axis=1)
            closest_indices = distances.nsmallest(10).index
            closest_rows = df.loc[closest_indices]
            
            predicted_disease = closest_rows[target_col].mode()[0]
            confidence = 0.5
            
            # Get top diseases from closest matches
            disease_counts = closest_rows[target_col].value_counts()
            top_diseases = {}
            for disease, count in disease_counts.head(3).items():
                disease_idx = disease_to_idx[disease]
                top_diseases[disease_idx] = count / len(closest_rows)
        
        # Display top 3 predictions
        print(f"\n🎯 Predictions:")
        for rank, (idx, conf) in enumerate(sorted(top_diseases.items(), key=lambda x: x[1], reverse=True)[:3], 1):
            disease = idx_to_disease[idx]
            bar = '█' * int(conf * 20)
            print(f"  {rank}. {disease}")
            print(f"     Confidence: {conf:.1%} {bar}")
    
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
            
            # Make prediction using dataset matching
            feature_vector_values = feature_vector.values
            
            # Find exact matches first
            matched_rows = df[df[feature_cols].eq(feature_vector_values).all(axis=1)]
            
            if len(matched_rows) > 0:
                disease_counts = matched_rows[target_col].value_counts()
                top_diseases = {}
                for disease, count in disease_counts.head(5).items():
                    disease_idx = disease_to_idx[disease]
                    top_diseases[disease_idx] = count / len(matched_rows)
            else:
                # Find closest matches
                distances = ((df[feature_cols] - feature_vector_values) ** 2).sum(axis=1)
                closest_indices = distances.nsmallest(20).index
                closest_rows = df.loc[closest_indices]
                
                disease_counts = closest_rows[target_col].value_counts()
                top_diseases = {}
                for disease, count in disease_counts.head(5).items():
                    disease_idx = disease_to_idx[disease]
                    top_diseases[disease_idx] = count / len(closest_rows)
            
            print(f"\n🎯 Top 5 Predicted Diseases:")
            for rank, (idx, conf) in enumerate(sorted(top_diseases.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                disease = idx_to_disease[idx]
                bar = '█' * int(conf * 30)
                print(f"  {rank}. {disease}")
                print(f"     {conf:.1%} {bar}")
            
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
        print("2. Run the 'Save Model Data' cell at the end of the notebook")
        print("3. Installed dependencies: pip install sentence-transformers pandas numpy")
