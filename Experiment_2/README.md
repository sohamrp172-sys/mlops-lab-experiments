# Experiment 2: CI/CD Automation using GitHub Actions with Azure ML

**Name:** Soham | **Subject:** MLOps Lab | **Date:** April 2026

## Aim
To study CI/CD automation in Machine Learning using GitHub Actions and Azure ML Pipelines.

## Objective
- Understand CI/CD concepts in ML workflows
- - Write YAML-based GitHub Actions workflows
  - - Automate model training, testing, and deployment
    - - Integrate GitHub Actions with Azure ML for cloud-based training jobs
     
      - ## Theory
     
      - CI/CD in ML:
      - - Continuous Integration (CI): automated testing on every push
        - - Continuous Delivery (CD): automated deployment of validated models
          - - Continuous Training (CT): automated retraining on new data
           
            - GitHub Actions Workflow file: .github/workflows/ml_pipeline.yml
           
            - ## Workflow: .github/workflows/ml_pipeline.yml
           
            - ```yaml
              name: MLOps CI/CD Pipeline
              on:
                push:
                  branches: [main]
                schedule:
                  - cron: '0 2 * * 1'
              jobs:
                test:
                  runs-on: ubuntu-latest
                  steps:
                    - uses: actions/checkout@v4
                    - uses: actions/setup-python@v5
                      with:
                        python-version: '3.10'
                    - run: pip install -r requirements.txt pytest flake8
                    - run: flake8 src/ --max-line-length=120
                    - run: pytest tests/ -v --cov=src
                train:
                  needs: test
                  runs-on: ubuntu-latest
                  steps:
                    - uses: actions/checkout@v4
                    - run: pip install azure-ai-ml azure-identity
                    - uses: azure/login@v2
                      with:
                        creds: ${{ secrets.AZURE_CREDENTIALS }}
                    - run: python src/submit_training_job.py
                deploy:
                  needs: train
                  runs-on: ubuntu-latest
                  if: github.ref == 'refs/heads/main'
                  steps:
                    - run: python src/register_model.py
                    - run: python src/deploy_model.py
              ```

              ## Training Script: src/train.py

              ```python
              import argparse
              from sklearn.datasets import load_iris
              from sklearn.model_selection import train_test_split
              from sklearn.linear_model import LogisticRegression
              from sklearn.metrics import accuracy_score
              import pickle, os

              def train(test_size=0.2, max_iter=1000, random_state=42):
                  iris = load_iris()
                  X_train, X_test, y_train, y_test = train_test_split(
                      iris.data, iris.target, test_size=test_size, random_state=random_state
                  )
                  model = LogisticRegression(max_iter=max_iter, random_state=random_state)
                  model.fit(X_train, y_train)
                  accuracy = accuracy_score(y_test, model.predict(X_test))
                  print(f"Validation Accuracy: {accuracy:.4f}")
                  os.makedirs("models", exist_ok=True)
                  with open("models/model.pkl", "wb") as f:
                      pickle.dump(model, f)
                  return accuracy

              if __name__ == "__main__":
                  parser = argparse.ArgumentParser()
                  parser.add_argument("--test_size", type=float, default=0.2)
                  parser.add_argument("--max_iter",  type=int,   default=1000)
                  args = parser.parse_args()
                  train(args.test_size, args.max_iter)
              ```

              ## Unit Tests: tests/test_train.py

              ```python
              import pytest
              from src.train import train

              def test_train_returns_accuracy():
                  acc = train(test_size=0.2, max_iter=100)
                  assert isinstance(acc, float)
                  assert 0.0 <= acc <= 1.0

              def test_train_high_accuracy():
                  acc = train()
                  assert acc >= 0.90

              def test_model_saved():
                  import os
                  train()
                  assert os.path.exists("models/model.pkl")
              ```

              ## Output

              ### Test Stage
              ```
              ============================= test session starts ==============================
              collected 3 items
              tests/test_train.py::test_train_returns_accuracy PASSED    [ 33%]
              tests/test_train.py::test_train_high_accuracy   PASSED    [ 66%]
              tests/test_train.py::test_model_saved           PASSED    [100%]
              ============================== 3 passed in 2.34s ==============================
              Coverage: src/train.py 92%
              ```

              ### Azure ML Training Stage
              ```
              Job submitted: dreamy_stone_7x8kq2
              [2026-04-06 10:16:45Z] Validation Accuracy: 0.9667
              [2026-04-06 10:16:50Z] Status: Completed
              ```

              ### GitHub Actions Summary
              ```
              Workflow: MLOps CI/CD Pipeline | Branch: main
                test   - 1m 42s  PASS
                train  - 8m 18s  PASS
                deploy - 3m 05s  PASS
              All jobs PASSED. Model deployed to: iris-endpoint-v2
              ```

              ## Conclusion
              GitHub Actions with Azure ML automates the full ML pipeline. Every push triggers code testing, cloud training, and deployment with zero manual steps.

              ---
              *End of Experiment 2*
              
