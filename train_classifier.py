import os
import argparse
import numpy as np
import torch
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from dataset import get_dataloaders
from model_dae import DAE

def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Extracting features"):
            data = data.to(device)
            latent = model.encode(data) 
            latent_flat = latent.view(latent.size(0), -1).cpu().numpy()
            features.append(latent_flat)
            labels.append(label.numpy())
    return np.vstack(features), np.concatenate(labels)

def train_classifiers(
    real_dir: str, fake_dir: str, model_path: str, device: str, batch_size: int, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    torch_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    print("1. Loading DAE model as feature extractor...")
    dae = DAE()
    if os.path.exists(model_path):
        dae.load_state_dict(torch.load(model_path, map_location=torch_device))
    else:
        print(f"Error: Pre-trained DAE not found at {model_path}")
        return
    dae.to(torch_device)

    print("2. Initializing DataLoaders...")
    train_real, test_real, test_fake = get_dataloaders(
        real_dir=real_dir, fake_dir=fake_dir, batch_size=batch_size, num_workers=4
    )

    print("\n3. Extracting Latent Vectors (Real Images)...")
    real_features_1, real_labels_1 = extract_features(dae, train_real, torch_device)
    real_features_2, real_labels_2 = extract_features(dae, test_real, torch_device)
    all_real_features = np.vstack([real_features_1, real_features_2])
    all_real_labels = np.concatenate([real_labels_1, real_labels_2])

    print("\n3. Extracting Latent Vectors (Fake Images)...")
    all_fake_features, all_fake_labels = extract_features(dae, test_fake, torch_device)

    print("\n4. Balancing dataset (1:1)...")
    min_samples = min(len(all_real_features), len(all_fake_features))
    np.random.seed(42)

    if len(all_fake_features) > min_samples:
        indices = np.random.choice(len(all_fake_features), min_samples, replace=False)
        all_fake_features = all_fake_features[indices]
        all_fake_labels = all_fake_labels[indices]
    
    if len(all_real_features) > min_samples:
        indices = np.random.choice(len(all_real_features), min_samples, replace=False)
        all_real_features = all_real_features[indices]
        all_real_labels = all_real_labels[indices]

    X = np.vstack([all_real_features, all_fake_features])
    y = np.concatenate([all_real_labels, all_fake_labels])
    print(f"Total balanced samples: {len(X)} (Features shape: {X.shape[1]})")

    print("\n5. Splitting into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n6. Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n7. Training Classifiers...")
    
    svm_clf = LinearSVC(max_iter=2000, random_state=42, dual=False)
    print(" - Training Linear SVM...")
    svm_clf.fit(X_train_scaled, y_train)
    
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    print(" - Training k-NN...")
    knn_clf.fit(X_train_scaled, y_train)

    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print(" - Training Random Forest...")
    rf_clf.fit(X_train_scaled, y_train)

    xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
    print(" - Training XGBoost...")
    xgb_clf.fit(X_train_scaled, y_train)

    print("\n8. Saving models and test datasets...")
    joblib.dump(svm_clf, os.path.join(output_dir, "svm_model.joblib"))
    joblib.dump(knn_clf, os.path.join(output_dir, "knn_model.joblib"))
    joblib.dump(rf_clf, os.path.join(output_dir, "rf_model.joblib"))
    joblib.dump(xgb_clf, os.path.join(output_dir, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    
    # Сохраняем тестовую выборку, чтобы не было утечки данных при оценке
    np.save(os.path.join(output_dir, "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print(f"All artifacts saved to '{output_dir}'. You can now run evaluate_classifier.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supervised Classifier on Latent Features")
    parser.add_argument("--real_dir", type=str, default="./data/processed/real")
    parser.add_argument("--fake_dir", type=str, default="./data/processed/fake")
    parser.add_argument("--model_path", type=str, default="./models/dae_model.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./models_supervised")
    args = parser.parse_args()

    train_classifiers(
        real_dir=args.real_dir, fake_dir=args.fake_dir, model_path=args.model_path,
        device=args.device, batch_size=args.batch_size, output_dir=args.output_dir,
    )