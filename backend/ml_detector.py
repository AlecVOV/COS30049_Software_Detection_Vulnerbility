import joblib
import numpy as np
import os

class VulnerabilityDetector:
    """
    Multi-model vulnerability detector using ensemble of 6 models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Naive Bayes
    """
    
    def __init__(self, models_dir='models'):
        """Load all trained models and vectorizer"""
        self.models_dir = models_dir
        
        print("🔒 Loading Reconnaise.ai Models...")
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found at {vectorizer_path}. "
                "Please run 'python train_model.py' first."
            )
        
        self.vectorizer = joblib.load(vectorizer_path)
        print("✓ Vectorizer loaded")
        
        # Load all models
        self.models = {}
        model_files = {
            'Logistic Regression': 'model_logistic_regression.joblib',
            'Random Forest': 'model_random_forest.joblib',
            'Gradient Boosting': 'model_gradient_boosting.joblib',
            'XGBoost': 'model_xgboost.joblib',
            'LightGBM': 'model_lightgbm.joblib',
            'Naive Bayes': 'model_naive_bayes.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
            else:
                print(f"⚠️  Warning: {model_name} not found at {model_path}")
        
        if not self.models:
            raise FileNotFoundError(
                "No models found. Please run 'python train_model.py' first."
            )
        
        print(f"\n✅ Loaded {len(self.models)} models successfully!\n")
    
    def predict(self, code):
        """
        Predict vulnerability using ensemble of all models
        
        Args:
            code (str): Source code to analyze
            
        Returns:
            dict: Prediction results with individual and ensemble predictions
        """
        # Vectorize code
        features = self.vectorizer.transform([code])
        
        # Get predictions from all models
        model_predictions = {}
        probabilities = []
        votes_vulnerable = 0
        
        for model_name, model in self.models.items():
            # Get probability of being vulnerable (class 1)
            proba = model.predict_proba(features)[0]
            vuln_probability = proba[1]
            
            is_vulnerable = vuln_probability > 0.5
            if is_vulnerable:
                votes_vulnerable += 1
            
            model_predictions[model_name] = {
                'vulnerable': is_vulnerable,
                'confidence': vuln_probability if is_vulnerable else (1 - vuln_probability),
                'safe_prob': proba[0],
                'vuln_prob': vuln_probability
            }
            
            probabilities.append(vuln_probability)
        
        # Ensemble prediction: average of all model probabilities
        avg_probability = np.mean(probabilities)
        
        # Weighted ensemble (give more weight to better models)
        # You can adjust weights based on model performance
        weights = {
            'XGBoost': 1,
            'LightGBM': 0.99,
            'Gradient Boosting': 0.99,
            'Random Forest': 0.98,
            'Logistic Regression': 0.99,
            'Naive Bayes': 0.92
        }
        
        weighted_probs = []
        total_weight = 0
        for model_name, prob in zip(self.models.keys(), probabilities):
            weight = weights.get(model_name, 1.0)
            weighted_probs.append(prob * weight)
            total_weight += weight
        
        weighted_avg_probability = sum(weighted_probs) / total_weight
        
        # Voting: majority wins
        votes_safe = len(probabilities) - votes_vulnerable
        
        # Final decision (using weighted average)
        is_vulnerable = weighted_avg_probability > 0.5
        
        return {
            'is_vulnerable': is_vulnerable,
            'confidence': weighted_avg_probability if is_vulnerable else (1 - weighted_avg_probability),
            'ensemble_probability': avg_probability,
            'weighted_probability': weighted_avg_probability,
            'voting': {
                'vulnerable': votes_vulnerable,
                'safe': votes_safe,
                'total_models': len(self.models),
                'decision': 'Vulnerable' if votes_vulnerable > votes_safe else 'Safe'
            },
            'model_predictions': model_predictions,
            'risk_level': self._get_risk_level(weighted_avg_probability)
        }
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        elif probability < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def predict_batch(self, code_list):
        """
        Predict vulnerabilities for multiple code samples
        
        Args:
            code_list (list): List of code strings
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(code) for code in code_list]
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'vectorizer_features': self.vectorizer.max_features
        }

# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TESTING VULNERABILITY DETECTOR")
    print("="*70)
    
    # Initialize detector
    try:
        detector = VulnerabilityDetector()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run: python train_model.py")
        exit(1)
    
    # Show model info
    info = detector.get_model_info()
    print(f"📊 Model Information:")
    print(f"  Total models: {info['total_models']}")
    print(f"  Vectorizer features: {info['vectorizer_features']}")
    
    # Test cases
    test_cases = [
        {
            'name': 'SQL Injection',
            'code': '''
def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()
'''
        },
        {
            'name': 'Command Injection',
            'code': '''
import os
def ping_server(host):
    command = "ping -c 1 " + host
    os.system(command)
'''
        },
        {
            'name': 'Safe Code',
            'code': '''
def calculate_sum(a, b):
    """Safely calculate sum of two numbers"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Arguments must be numbers")
    return a + b
'''
        }
    ]
    
    # Test each case
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['name']}")
        print(f"{'='*70}")
        
        result = detector.predict(test['code'])
        
        # Overall result
        if result['is_vulnerable']:
            print(f"\n🔴 VERDICT: VULNERABLE")
        else:
            print(f"\n🟢 VERDICT: SAFE")
        
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Ensemble Probability: {result['ensemble_probability']:.1%}")
        print(f"   Weighted Probability: {result['weighted_probability']:.1%}")
        
        # Voting results
        print(f"\n📊 Voting Results:")
        print(f"   Vulnerable: {result['voting']['vulnerable']}/{result['voting']['total_models']}")
        print(f"   Safe: {result['voting']['safe']}/{result['voting']['total_models']}")
        print(f"   Decision: {result['voting']['decision']}")
        
        # Individual predictions
        print(f"\n🤖 Individual Model Predictions:")
        for model_name, pred in result['model_predictions'].items():
            status = "🔴 VULN" if pred['vulnerable'] else "🟢 SAFE"
            bar_length = int(pred['vuln_prob'] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {model_name:20s}: {status} {bar} {pred['vuln_prob']:.1%}")
    
    print(f"\n{'='*70}")
    print("✅ TESTING COMPLETE")
    print(f"{'='*70}")