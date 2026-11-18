import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import FEATURES, TARGET

def train_models():
    df1 = pd.read_csv("data/processed/conveyor_v1_features.csv")
    df2 = pd.read_csv("data/processed/conveyor_v2_features.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Conveyor_Jam_Prediction")
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        )
    }
    results, best_model, best_name, best_score = [], None, None, -1
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            cm = confusion_matrix(y_test, predictions)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset", "v1_and_v2")
            mlflow.log_param("test_size", 0.25)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
            results.append({
                "model": model_name, "accuracy": accuracy,
                "recall": recall, "f1_score": f1,
                "confusion_matrix": cm.tolist()
            })
            print(f"\nModel: {model_name}\nAccuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\nConfusion Matrix:\n{cm}")
            if f1 > best_score:
                best_score, best_model, best_name = f1, model, model_name
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("reports/model_comparison.csv", index=False)
    print("\nSelected model:", best_name)
    with mlflow.start_run(run_name="Selected_Model_Registration"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_f1_score", best_score)
        mlflow.sklearn.log_model(
            best_model, artifact_path="selected_model",
            registered_model_name="ConveyorJamPredictor"
        )
    print("Model registered in MLflow.")
    print("Saved model to models/best_model.joblib")
    return pd.DataFrame(results)

if __name__ == "__main__":
    train_models()
