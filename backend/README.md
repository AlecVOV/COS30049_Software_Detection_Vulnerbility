# SecureCode AI - PHP Vulnerability Detection System

A comprehensive vulnerability detection system for PHP code using ensemble machine learning and rule-based analysis.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training Models](#-training-models)
- [Running the Application](#-running-the-application)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Testing the Detector](#-testing-the-detector)
- [Troubleshooting](#-troubleshooting)

## 🔍 Overview

SecureCode AI is a hybrid vulnerability detection system that combines:
- **6 Machine Learning Models** (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, Naive Bayes)
- **Rule-Based Detection** for common PHP vulnerabilities
- **Ensemble Predictions** with weighted voting

The system detects various PHP vulnerabilities including SQL Injection, XSS, Command Injection, Path Traversal, and more.

## ✨ Features

- 🤖 Ensemble of 6 trained ML models
- 📊 Interactive Streamlit web interface
- 🔍 Real-time code scanning
- 📈 Comprehensive model comparison
- 📉 Performance visualizations (ROC curves, Precision-Recall curves, Confusion matrices)
- 🎯 Rule-based vulnerability detection
- 📊 Dataset analysis dashboard

## 📁 Project Structure

```
├── app.py                              # Streamlit web application
├── ml_detector.py                      # ML-based vulnerability detector
├── vuln_rules.py                       # Rule-based detector
├── train_model.py                      # Model training script
├── chunk_dataset.py                    # Dataset chunking utility
├── clean_merge_chunk_dataset.ipynb     # Dataset preprocessing notebook
├── model_comparison.ipynb              # Model evaluation notebook
├── requirements.txt                    # Python dependencies
├── data/
│   ├── balanced_merged_dataset.csv     # Final balanced dataset for training
│   ├── merged_all_datasets.csv         # Merged unbalanced dataset
│   ├── dataset_1_chunking_results.csv  # Dataset 1 (DiverseVul - PHP filtered)
│   ├── dataset_2_chunking_results.csv  # Dataset 2
│   ├── dataset_3_chunking_results.csv  # Dataset 3
│   ├── Dataset_1/                      # Raw dataset samples
│   ├── Dataset_2/                      # Raw dataset samples
│   └── Dataset_3/                      # Raw dataset samples
│
├── models/                             # Trained models directory
│   ├── model_logistic_regression.joblib
│   ├── model_random_forest.joblib
│   ├── model_gradient_boosting.joblib
│   ├── model_xgboost.joblib
│   ├── model_lightgbm.joblib
│   ├── model_naive_bayes.joblib
│   └── vectorizer.joblib
└── docs/                               # Documentation and visualizations
    ├── model_comparison.csv
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── accuracy_vs_speed.png
    └── confusion_matrix_*.png
```

## 🚀 Installation

### Step 1: Clone or Extract the Project

```powershell
cd "C:\Users\Admin\Desktop\Vọc Code - OnlyPHP"
```

### Step 2: Create Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Required packages include:**
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- joblib

## 📊 Dataset Preparation

The project uses three datasets that are merged and balanced:

### Dataset Sources:
1. **Dataset 1**: DiverseVul (filtered for PHP code only) - 7,665 samples
2. **Dataset 2**: Custom PHP vulnerability dataset - 42,204 samples
3. **Dataset 3**: Merged vulnerability datasets - 13,000 samples

### Pre-processed Datasets Available:

If you have the pre-processed datasets in `data/`, you can skip to [Training Models](#-training-models).

### Manual Dataset Processing (Optional):

If you need to regenerate the datasets or make modifications, follow these steps:

#### 0. Convert all PHP code into CSV file fore Dataset_3:
```powershell
# Run the Jupyter notebook for dataset cleaning and merging
jupyter notebook write_php_to_csv.ipynb
```
This notebook extracts PHP code snippets from raw files and saves them into `[folder_name]_dataset.csv`.
-If you need to regenerate the datasets:
 --- IGNORE ---

Then you can merge all the `[folder_name]_dataset.csv` into dataset_3_chunking_results.csv by run the following command in the Dataset_3 folder:
```powershell
python merging_dataset.py
```

#### 1. Process Individual Datasets:

```powershell
# Run the Jupyter notebook for dataset cleaning and merging
jupyter notebook clean_merge_chunk_dataset.ipynb
```

This notebook performs:
- Filtering PHP code from DiverseVul dataset
- Renaming columns to standardized format (`code`, `is_vulnerable`)
- Merging all datasets
- Balancing vulnerable and safe samples

#### 2. Alternative: Use Chunking Script:

```powershell
python chunk_dataset.py
```

### Final Dataset Structure:

After processing, you should have:
- `data/balanced_merged_dataset.csv` - **47,858 samples** (balanced)
- `data/merged_all_datasets.csv` - All data merged (unbalanced)

**Dataset Format:**
```csv
code,is_vulnerable
"<?php echo $_GET['name']; ?>",1
"<?php echo htmlspecialchars($_GET['name']); ?>",0
```

## 🎓 Training Models

### Step 1: Ensure Dataset is Ready

Verify that `data/balanced_merged_dataset.csv` exists and contains balanced data.

### Step 2: Train All Models

```powershell
python train_model.py
```

**Training Process:**
1. Loads `balanced_merged_dataset.csv`
2. Splits data (80% train, 20% test)
3. Creates TF-IDF vectorizer (3000 features, 1-2 grams)
4. Trains 6 models sequentially:
   - Logistic Regression
   - Random Forest (100 trees)
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - Naive Bayes
5. Evaluates each model (accuracy, precision, recall, F1-score)
6. Generates visualizations (confusion matrices, ROC curves, PR curves)
7. Saves models to `models/` directory

**Expected Output:**
```
📊 Results:
  Accuracy:  0.9876 (98.76%)
  Precision: 0.9850
  Recall:    0.9902
  F1-Score:  0.9876
```

**Generated Files:**
- `models/model_*.joblib` - 6 trained models
- `models/vectorizer.joblib` - TF-IDF vectorizer
- `docs/model_comparison.csv` - Performance metrics
- `docs/*.png` - Visualization charts

### Step 3: Verify Training and Export Charts

```powershell
# Run model comparison notebook to export some charts for later reference
jupyter notebook model_comparison.ipynb
```

## 🌐 Running the Application

### Start the Streamlit App:

```powershell
streamlit run app.py
```

**or**

```powershell
python -m streamlit run app.py
```

The application will open in your browser at: **http://localhost:8501**

### Application Pages:

1. **🏠 Home** - Overview and system information
2. **📊 Data Analysis** - Dataset statistics and distribution
3. **🔍 Scan Code** - Upload or paste PHP code for vulnerability detection
4. **📈 Model Comparison** - Performance metrics and model evaluation

## 💻 Usage

### 1. Home Page

View system overview, features, and quick stats.

### 2. Data Analysis

- View dataset distribution (vulnerable vs safe)
- Analyze dataset sources
- Explore sample code snippets

### 3. Scan Code

**Upload File:**
```php
<?php
// Upload a .php file
$username = $_GET['user'];
$query = "SELECT * FROM users WHERE name='$username'";
```

**Or Paste Code:**
```php
<?php
$user_input = $_POST['data'];
eval($user_input);
?>
```

**Results Include:**
- 🔴/🟢 Vulnerability verdict
- Confidence score and risk level
- Ensemble prediction (weighted average)
- Individual model predictions with probabilities
- Rule-based detection results
- Specific vulnerability types detected

### 4. Model Comparison

View comprehensive model performance:
- Accuracy, Precision, Recall, F1-Score
- Training time comparison
- ROC curves (AUC scores)
- Precision-Recall curves
- Confusion matrices for each model
- Model selection justification

## 🤖 Model Information

### Ensemble Models:

| Model | Description | Accuracy* |
|-------|-------------|-----------|
| **Logistic Regression** | Linear classifier with L2 regularization | ~98.8% |
| **Random Forest** | Ensemble of 100 decision trees | ~99.8% |
| **Gradient Boosting** | Sequential boosting with decision trees | ~99.8% |
| **XGBoost** | Optimized gradient boosting | ~99.9% |
| **LightGBM** | Fast gradient boosting | ~99.8% |
| **Naive Bayes** | Probabilistic classifier | ~92.3% |

*Approximate values - see `docs/model_comparison.csv` for exact metrics

### Weighted Ensemble:

The `VulnerabilityDetector` class in `ml_detector.py` uses weighted voting:
- XGBoost: 1.0 (highest weight)
- LightGBM: 0.99
- Gradient Boosting: 0.99
- Random Forest: 0.98
- Logistic Regression: 0.99
- Naive Bayes: 0.92

### Vulnerability Detection Rules:

The `RuleBasedDetector` in `vuln_rules.py` detects:
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Path Traversal
- Code Injection
- File Inclusion
- Insecure Deserialization
- XML External Entity (XXE)
- Server-Side Request Forgery (SSRF)
- And more...

## 📈 Testing the Detector

### Using ML Detector Directly:

```python
from ml_detector import VulnerabilityDetector

detector = VulnerabilityDetector()

code = """
<?php
$username = $_GET['user'];
$query = "SELECT * FROM users WHERE name='$username'";
mysqli_query($conn, $query);
?>
"""

result = detector.predict(code)
print(f"Vulnerable: {result['is_vulnerable']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Run Built-in Tests:

```powershell
python ml_detector.py
```

This will run test cases including:
- SQL Injection detection
- Command Injection detection
- Safe code validation

## 🔧 Troubleshooting

### Issue: Models not found

**Error:** `FileNotFoundError: No models found`

**Solution:**
```powershell
python train_model.py
```

### Issue: Dataset not found

**Error:** `FileNotFoundError: data/balanced_merged_dataset.csv not found`

**Solution:**
1. Check if datasets exist in `data/`
2. Run `clean_merge_chunk_dataset.ipynb` to generate datasets
3. Or run `chunk_dataset.py`

### Issue: Streamlit not starting

**Solution:**
```powershell
pip install --upgrade streamlit
streamlit run app.py
```

### Issue: PowerShell execution policy error

**Error:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Low model accuracy

**Solution:**
- Ensure you're using `balanced_merged_dataset.csv`
- Verify dataset has ~47,858 balanced samples
- Retrain models with `train_model.py`

### Issue: Import errors

**Solution:**
```powershell
# Reinstall all dependencies
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📝 Additional Notes

- **Training Time**: Full training takes ~5-15 minutes depending on hardware
- **Model Size**: All models combined ~100-200 MB
- **Memory Usage**: ~2-4 GB RAM during training
- **Best Model**: XGBoost typically achieves highest accuracy (~99.9%)
- **Dataset Balance**: Training uses balanced dataset (50% vulnerable, 50% safe)

## 📚 References

- **Dataset Source**: DiverseVul (filtered for PHP)
- **Models**: scikit-learn, XGBoost, LightGBM
- **Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn

## 🎯 Quick Command Reference

```powershell
# 1. Setup environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (one-time, takes 5-15 minutes)
python train_model.py

# 4. Run application
streamlit run app.py

# 5. Test detector
python ml_detector.py

# 6. Open Jupyter notebooks (optional)
jupyter notebook clean_merge_chunk_dataset.ipynb
jupyter notebook model_comparison.ipynb
```

## 📖 Step-by-Step: First Time Setup

### Complete Workflow:

```powershell
# Step 1: Navigate to project
cd "C:\Users\Admin\Desktop\Vọc Code - OnlyPHP"

# Step 2: Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Train models (if not already trained)
python train_model.py

# Step 5: Run the application
streamlit run app.py
```

### If Training Fails (Missing Dataset):

```powershell
# Option 1: Use Jupyter notebook to prepare datasets
jupyter notebook clean_merge_chunk_dataset.ipynb
# Follow notebook instructions, then run:
python train_model.py

# Option 2: Use chunking script
python chunk_dataset.py
python train_model.py
```

