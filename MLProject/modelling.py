import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import matplotlib.pyplot as plt
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ========================
# Setup MLflow
# ========================
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', '')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# ========================
# Argument Parser
# ========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=5)
args = parser.parse_args()

# ========================
# Load Data
# ========================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'ObesityDataSet_preprocessing.csv')
df = pd.read_csv(csv_path)

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# Training dengan active run dari MLflow Project
# ========================
with mlflow.start_run(nested=True):

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Manual logging
    mlflow.log_param('n_estimators', args.n_estimators)
    mlflow.log_param('max_depth', args.max_depth)
    mlflow.log_param('min_samples_split', args.min_samples_split)
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall', rec)
    mlflow.log_metric('f1_score', f1)

    # Artefak
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - CI Pipeline')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    report = classification_report(y_test, y_pred, output_dict=True)
    with open('classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact('classification_report.json')

    mlflow.sklearn.log_model(model, 'model')

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Simpan run ID untuk Docker build
    run_id = mlflow.active_run().info.run_id
    with open('run_id.txt', 'w') as f:
        f.write(run_id)
    print(f"Run ID: {run_id}")