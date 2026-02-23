import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import urllib.request

# Configuration
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
STATIC_PLOTS = os.path.join(os.path.dirname(__file__), '..', 'static', 'plots')

# Create directories if not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_PLOTS, exist_ok=True)

def download_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Dataset downloaded.")
    return pd.read_csv(DATA_PATH)

def train_models():
    print("Loading Data...")
    df = download_data()

    # Preprocessing: Handle zeros as missing values for specific columns
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col] = df[col].replace(0, df[col].mean())

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        # Cross Validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results.append({
            'Model': name,
            'Accuracy': f"{acc:.4f}",
            'CV Mean': f"{cv_scores.mean():.4f}",
            'CV Std': f"{cv_scores.std():.4f}"
        })

        print(f"Accuracy: {acc:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # Save Comparison Table (to be displayed in README)
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))

    # Save Model and Scaler
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f"\nBest Model: {best_model_name} saved successfully.")

    # Visualization 1: ROC Curve
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(STATIC_PLOTS, 'roc_curve.png'))
    print("ROC Curve saved.")
    plt.close()

    # Visualization 2: Feature Importance (Only for Random Forest/Tree models)
    if isinstance(best_model, RandomForestClassifier):
        feature_names = X.columns
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance (Random Forest)")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_PLOTS, 'feature_importance.png'))
        print("Feature Importance saved.")
        plt.close()

if __name__ == "__main__":
    train_models()