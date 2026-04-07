# Experiment 6: Tracking ML Experiments using MLflow with Azure ML

**Name:** Soham | **Subject:** MLOps Lab | **Date:** April 2026

## Aim
To study ML experiment tracking using MLflow and its integration with Azure Machine Learning.

## Objective
- Log parameters, metrics, and model artifacts for multiple ML models
- Compare experiment runs in the MLflow UI
- Register the best model in MLflow Model Registry
- Integrate MLflow with Azure ML workspace for centralized tracking

## Theory

Without tracking: results are forgotten, experiments cannot be reproduced, best models are unclear.

MLflow Components:
- Tracking: log params, metrics, artifacts per run
- Models: standard model packaging format
- Registry: version models with lifecycle stages (None, Staging, Production)

Key MLflow APIs:
- mlflow.start_run() - starts a new run
- mlflow.log_param(key, value) - log a hyperparameter
- mlflow.log_metric(key, value) - log an evaluation metric
- mlflow.log_artifact(path) - save a file to the run
- mlflow.sklearn.log_model(model, \"model\") - log and optionally register a model

## Code: experiment_tracking.py

    import mlflow, mlflow.sklearn
        import matplotlib.pyplot as plt, seaborn as sns
            from sklearn.datasets import load_iris
                from sklearn.model_selection import train_test_split, cross_val_score
                    from sklearn.linear_model import LogisticRegression
                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                            from sklearn.svm import SVC
                                from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                                    import os, warnings
                                        warnings.filterwarnings(\"ignore\")

                                            mlflow.set_tracking_uri(\"mlruns\")
                                                mlflow.set_experiment(\"iris-classification-comparison\")

                                                    iris = load_iris()
                                                        X, y = iris.data, iris.target
                                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                                                                models = {
                                                                        \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),
                                                                                \"RandomForest\":       RandomForestClassifier(n_estimators=100, random_state=42),
                                                                                        \"GradientBoosting\":   GradientBoostingClassifier(n_estimators=100, random_state=42),
                                                                                                \"SVM\":                SVC(probability=True, random_state=42),
                                                                                                    }
                                                                                                    
                                                                                                        for name, model in models.items():
                                                                                                                with mlflow.start_run(run_name=name) as run:
                                                                                                                            mlflow.set_tag(\"model_type\", name)
                                                                                                                                        mlflow.set_tag(\"dataset\", \"iris\")
                                                                                                                                        
                                                                                                                                                    model.fit(X_train, y_train)
                                                                                                                                                                y_pred = model.predict(X_test)
                                                                                                                                                                
                                                                                                                                                                            accuracy = accuracy_score(y_test, y_pred)
                                                                                                                                                                                        f1       = f1_score(y_test, y_pred, average=\"weighted\")
                                                                                                                                                                                                    cv_mean  = cross_val_score(model, X, y, cv=5).mean()
                                                                                                                                                                                                    
                                                                                                                                                                                                                mlflow.log_metric(\"accuracy\", accuracy)
                                                                                                                                                                                                                            mlflow.log_metric(\"f1_score\", f1)
                                                                                                                                                                                                                                        mlflow.log_metric(\"cv_mean\",  cv_mean)
                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                    os.makedirs(\"artifacts\", exist_ok=True)
                                                                                                                                                                                                                                                                cm = confusion_matrix(y_test, y_pred)
                                                                                                                                                                                                                                                                            fig, ax = plt.subplots(figsize=(5, 4))
                                                                                                                                                                                                                                                                                        sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\",
                                                                                                                                                                                                                                                                                                                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
                                                                                                                                                                                                                                                                                                                            ax.set_title(f\"{name} Confusion Matrix\")
                                                                                                                                                                                                                                                                                                                                        plt.tight_layout()
                                                                                                                                                                                                                                                                                                                                                    cm_path = f\"artifacts/{name}_cm.png\"
                                                                                                                                                                                                                                                                                                                                                                plt.savefig(cm_path); plt.close()
                                                                                                                                                                                                                                                                                                                                                                            mlflow.log_artifact(cm_path)
                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                        mlflow.sklearn.log_model(model, \"model\",
                                                                                                                                                                                                                                                                                                                                                                                                                             registered_model_name=f\"iris-{name.lower()}\")
                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                         print(f\"{name:20s} | acc={accuracy:.4f} | f1={f1:.4f} | cv={cv_mean:.4f}\")
                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                         ## Run Commands
                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                             pip install mlflow scikit-learn matplotlib seaborn
                                                                                                                                                                                                                                                                                                                                                                                                                                                 python experiment_tracking.py
                                                                                                                                                                                                                                                                                                                                                                                                                                                     mlflow ui --host 0.0.0.0 --port 5000
                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                     ## Output
                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                         LogisticRegression   | acc=0.9667 | f1=0.9666 | cv=0.9733
                                                                                                                                                                                                                                                                                                                                                                                                                                                             RandomForest         | acc=1.0000 | f1=1.0000 | cv=0.9600
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 GradientBoosting     | acc=1.0000 | f1=1.0000 | cv=0.9600
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     SVM                  | acc=1.0000 | f1=1.0000 | cv=0.9733
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ## Conclusion
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     MLflow tracked 4 classifiers with complete experiment history. Model Registry lifecycle management makes MLflow essential for team-scale MLOps experiment governance and model deployment decisions.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
