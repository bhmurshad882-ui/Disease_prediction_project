import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os

def main():
    print("=" * 60)
    print("  Disease Prediction ANN - Training Pipeline")
    print("=" * 60)

    # ── Load Data ──────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Blood_samples_dataset_balanced_2(f).csv')
    df = pd.read_csv(csv_path)
    print(f"      Loaded {df.shape[0]} samples with {df.shape[1] - 1} features")
    print(f"      Classes: {df['Disease'].unique().tolist()}")
    print(f"      Distribution:\n{df['Disease'].value_counts().to_string()}")

    # ── Preprocessing ─────────────────────────────────────────
    print("\n[2/5] Preprocessing...")
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    feature_names = X.columns.tolist()

    # Encode target labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = encoder.classes_.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"      Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── Build & Train ANN ─────────────────────────────────────
    print("\n[3/5] Training ANN (MLPClassifier)...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        verbose=True
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    print("\n[4/5] Evaluating model...")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    print(f"\n      Train Accuracy: {train_acc:.4f}")
    print(f"      Test  Accuracy: {test_acc:.4f}")
    print(f"\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print(f"      Confusion Matrix:\n{cm}")

    # ── Save Artifacts ────────────────────────────────────────
    print("\n[5/5] Saving artifacts...")
    model_dir = os.path.dirname(__file__)

    joblib.dump(model, os.path.join(model_dir, 'disease_ann_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(model_dir, 'encoder.pkl'))

    # Save metadata for frontend usage
    metadata = {
        "feature_names": feature_names,
        "class_names": class_names,
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "num_samples": int(df.shape[0]),
        "num_features": int(len(feature_names)),
        "architecture": "128 → 64 → 32 (ReLU, Adam)",
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("      [SUCCESS] Saved: disease_ann_model.pkl")
    print("      [SUCCESS] Saved: scaler.pkl")
    print("      [SUCCESS] Saved: encoder.pkl")
    print("      [SUCCESS] Saved: model_metadata.json")
    print("\n" + "=" * 60)
    print(f"  Training Complete - Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
