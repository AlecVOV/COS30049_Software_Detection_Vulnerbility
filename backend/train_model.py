import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,          
    auc,                
    precision_recall_curve  
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import re   
from collections import Counter  

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('docs', exist_ok=True)

print("="*70)
print("SECUECODE AI - MODEL TRAINING")
print("="*70)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('data/balanced_merged_dataset.csv')
print(f"✓ Total samples: {len(df):,}")
print(f"  Vulnerable: {sum(df['is_vulnerable'] == 1):,}")
print(f"  Safe: {sum(df['is_vulnerable'] == 0):,}")

# Show dataset sources if available
if 'source' in df.columns:
    print("\n📊 Dataset sources:")
    for source in df['source'].unique():
        count = sum(df['source'] == source)
        vuln = sum((df['source'] == source) & (df['is_vulnerable'] == 1))
        safe = count - vuln
        print(f"  {source}: {count:,} samples ({vuln:,} vuln, {safe:,} safe)")

# Prepare data
X = df['code']
y = df['is_vulnerable']

# Split data
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train):,}")
print(f"  Testing samples: {len(X_test):,}")

# Vectorize text
print("\n🔤 Vectorizing code samples...")
vectorizer = TfidfVectorizer(
    max_features=3000,      # Reduced for faster training
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Vectorization complete")
print(f"  Feature dimensions: {X_train_vec.shape[1]:,}")

# Save vectorizer
joblib.dump(vectorizer, 'models/vectorizer.joblib')
print("✓ Vectorizer saved to models/vectorizer.joblib")

# Define all 6 models
print("\n" + "="*70)
print("DEFINING MODELS")
print("="*70)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        'description': 'Linear classifier with L2 regularization'
    },
    
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        'description': 'Ensemble of 100 decision trees'
    },
    
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        ),
        'description': 'Sequential boosting with decision trees'
    },
    
    'XGBoost': {
        'model': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            min_child_weight=1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        ),
        'description': 'Optimized gradient boosting (XGBoost)'
    },
    
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'description': 'Fast gradient boosting (LightGBM)'
    },
    
    'Naive Bayes': {
        'model': MultinomialNB(
            alpha=1.0,
            fit_prior=True
        ),
        'description': 'Probabilistic classifier with Laplace smoothing'
    }
}

print(f"\n✓ {len(models)} models defined:")
for name, info in models.items():
    print(f"  • {name}: {info['description']}")

# Store results
results = {}
training_times = {}

# Train and evaluate each model
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

for model_name, model_info in models.items():
    print(f"\n{'='*70}")
    print(f"🤖 Training: {model_name}")
    print(f"{'='*70}")
    print(f"Description: {model_info['description']}")
    
    model = model_info['model']
    
    # Train model with timing
    print(f"\n⏱️  Training started...")
    start_time = time.time()
    
    model.fit(X_train_vec, y_train)
    
    training_time = time.time() - start_time
    training_times[model_name] = training_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print(f"🔍 Evaluating on test set...")
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n{'-'*70}")
    print("Classification Report:")
    print(f"{'-'*70}")
    print(classification_report(y_test, y_pred, 
          target_names=['Safe', 'Vulnerable'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{'-'*70}")
    print("Confusion Matrix:")
    print(f"{'-'*70}")
    print(f"                 Predicted")
    print(f"                Safe  Vulnerable")
    print(f"Actual Safe     {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"       Vuln     {cm[1][0]:>5}  {cm[1][1]:>5}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Vulnerable'],
                yticklabels=['Safe', 'Vulnerable'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save plot
    filename = model_name.lower().replace(' ', '_')
    plt.savefig(f'docs/confusion_matrix_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to docs/confusion_matrix_{filename}.png")
    
    # Save model
    model_filename = f'models/model_{filename}.joblib'
    joblib.dump(model, model_filename)
    print(f"✓ Model saved to {model_filename}")

# Compare all models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1_score'] for r in results.values()],
    'Training Time (s)': [r['training_time'] for r in results.values()]
})

# Sort by accuracy
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Find best models
best_accuracy_model = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']
fastest_model = comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model']
fastest_time = comparison_df['Training Time (s)'].min()

print(f"\n🏆 Best Accuracy: {best_accuracy_model} ({best_accuracy:.4f})")
print(f"⚡ Fastest Training: {fastest_model} ({fastest_time:.2f}s)")

# Save comparison results
comparison_df.to_csv('docs/model_comparison.csv', index=False)
print(f"✓ Comparison results saved to docs/model_comparison.csv")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n📈 Performance Range:")
print(f"  Accuracy:  {comparison_df['Accuracy'].min():.4f} - {comparison_df['Accuracy'].max():.4f}")
print(f"  Precision: {comparison_df['Precision'].min():.4f} - {comparison_df['Precision'].max():.4f}")
print(f"  Recall:    {comparison_df['Recall'].min():.4f} - {comparison_df['Recall'].max():.4f}")
print(f"  F1-Score:  {comparison_df['F1-Score'].min():.4f} - {comparison_df['F1-Score'].max():.4f}")

print(f"\n⏱️  Training Time Range:")
print(f"  Fastest: {comparison_df['Training Time (s)'].min():.2f}s")
print(f"  Slowest: {comparison_df['Training Time (s)'].max():.2f}s")
print(f"  Average: {comparison_df['Training Time (s)'].mean():.2f}s")
print(f"  Total:   {comparison_df['Training Time (s)'].sum():.2f}s")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Models saved in 'models/' directory:")
for name in results.keys():
    filename = name.lower().replace(' ', '_')
    print(f"  ✓ model_{filename}.joblib")
print(f"  ✓ vectorizer.joblib")

print(f"\nComparison results saved in 'docs/' directory:")
print(f"  ✓ model_comparison.csv")