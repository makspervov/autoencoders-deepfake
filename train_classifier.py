import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from dataset import get_dataloaders
from model_dae_classifier import DAE

def extract_features(model, dataloader, device):
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Extracting features"):
            data = data.to(device)
            latent = model.encoder(data) 
            latent_flat = latent.view(latent.size(0), -1).cpu().numpy()
            
            features.append(latent_flat)
            labels.append(label.numpy())
            
    return np.vstack(features), np.concatenate(labels)

def train_and_evaluate_classifier(
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

    print("\n8. Evaluating Supervised Models...")
    
    def evaluate_model(name, clf, X_test, y_test):
        if hasattr(clf, "decision_function"):
            y_scores = clf.decision_function(X_test)
            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        else:
            y_scores = clf.predict_proba(X_test)[:, 1]

        y_pred = clf.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        
        print(f"\n--- {name} Results ---")
        print(f"AUC Score: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return fpr, tpr, roc_auc

    fpr_svm, tpr_svm, auc_svm = evaluate_model("Support Vector Machine (LinearSVC)", svm_clf, X_test_scaled, y_test)
    fpr_knn, tpr_knn, auc_knn = evaluate_model("k-Nearest Neighbors (k=5)", knn_clf, X_test_scaled, y_test)
    fpr_rf, tpr_rf, auc_rf = evaluate_model("Random Forest (n=200)", rf_clf, X_test_scaled, y_test)
    fpr_xgb, tpr_xgb, auc_xgb = evaluate_model("XGBoost (n=200)", xgb_clf, X_test_scaled, y_test)

    # 9. Рисуем совместную ROC-кривую
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_svm, tpr_svm, color="darkorange", lw=2, label=f"SVM (AUC = {auc_svm:.3f})")
    plt.plot(fpr_knn, tpr_knn, color="green", lw=2, label=f"k-NN (AUC = {auc_knn:.3f})")
    plt.plot(fpr_rf, tpr_rf, color="red", lw=2, label=f"Random Forest (AUC = {auc_rf:.3f})")
    plt.plot(fpr_xgb, tpr_xgb, color="purple", lw=2, label=f"XGBoost (AUC = {auc_xgb:.3f})")
    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Supervised Classification ROC Curves on DAE Latent Features")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "supervised_roc_curves.png"))
    plt.close()
    
    print(f"\nROC curves saved to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supervised Classifier on Latent Features")
    parser.add_argument("--real_dir", type=str, default="./data/processed2/real")
    parser.add_argument("--fake_dir", type=str, default="./data/processed2/fake")
    parser.add_argument("--model_path", type=str, default="./models/dae_model.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./results/ClassifiedDAE")

    args = parser.parse_args()

    train_and_evaluate_classifier(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )