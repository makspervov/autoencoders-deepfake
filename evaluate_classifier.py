import os
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

def evaluate_classifiers(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"1. Loading test dataset from '{input_dir}'...")
    try:
        X_test_scaled = np.load(os.path.join(input_dir, "X_test_scaled.npy"))
        y_test = np.load(os.path.join(input_dir, "y_test.npy"))
    except FileNotFoundError:
        print("Error: Test dataset not found. Run train_classifier.py first.")
        return

    print("2. Loading trained models...")
    try:
        svm_clf = joblib.load(os.path.join(input_dir, "svm_model.joblib"))
        knn_clf = joblib.load(os.path.join(input_dir, "knn_model.joblib"))
        rf_clf = joblib.load(os.path.join(input_dir, "rf_model.joblib"))
        xgb_clf = joblib.load(os.path.join(input_dir, "xgb_model.joblib"))
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    print("\n3. Evaluating Supervised Models...")
    
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

    print("\n4. Plotting ROC Curves...")
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
    
    plot_path = os.path.join(output_dir, "supervised_roc_curves.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nEvaluation complete. ROC curves saved to '{plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Supervised Classifiers")
    parser.add_argument("--input_dir", type=str, default="./models_supervised", help="Directory with saved models and datasets")
    parser.add_argument("--output_dir", type=str, default="./results_supervised", help="Directory to save plots")
    args = parser.parse_args()

    evaluate_classifiers(input_dir=args.input_dir, output_dir=args.output_dir)