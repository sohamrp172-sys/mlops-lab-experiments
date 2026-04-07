# Experiment 2: Automating ML Training using GitHub Actions with Azure ML

**Name:** Soham | **Subject:** MLOps Lab | **Date:** April 2026

## Aim
To study CI/CD automation in ML using GitHub Actions and Azure ML Pipelines.

## Objective
- Understand CI/CD concepts in ML workflows
- Write YAML-based GitHub Actions workflows
- Automate model training, testing, and deployment
- Integrate GitHub Actions with Azure ML

## Theory
CI/CD in ML: Continuous Integration automates testing on every push, Continuous Delivery automates deployment of validated models, Continuous Training retrains on new data.

GitHub Actions uses .github/workflows/*.yml files. Core concepts: Workflow, Event (push/PR/schedule), Job, Step, Runner, Secret.

## Key Code: .github/workflows/ml_pipeline.yml
```yaml
name: MLOps CI/CD Pipeline
on:
  push:
      branches: [main]
      jobs:
        test:
            runs-on: ubuntu-latest
                steps:
                      - uses: actions/checkout@v4
                            - uses: actions/setup-python@v5
                                    with: {python-version: '3.10'}
                                          - run: pip install -r requirements.txt pytest
                                                - run: pytest tests/ -v
                                                  train:
                                                      needs: test
                                                          runs-on: ubuntu-latest
                                                              steps:
                                                                    - uses: actions/checkout@v4
                                                                          - run: pip install azure-ai-ml azure-identity
                                                                                - uses: azure/login@v2
                                                                                        with: {creds: '${{ secrets.AZURE_CREDENTIALS }}'}
                                                                                              - run: python src/submit_training_job.py
                                                                                              ```

                                                                                              ## Output
                                                                                              ```
                                                                                              Run pytest tests/ -v
                                                                                              ============================= test session starts ==============================
                                                                                              collected 3 items
                                                                                              tests/test_train.py::test_train_returns_accuracy PASSED    [ 33%]
                                                                                              tests/test_train.py::test_train_high_accuracy PASSED       [ 66%]
                                                                                              tests/test_train.py::test_model_saved PASSED               [100%]
                                                                                              ============================== 3 passed in 2.34s ==============================

                                                                                              Job submitted: dreamy_stone_7x8kq2
                                                                                              Job status: Running
                                                                                              [2026-04-06 10:16:45Z] Validation Accuracy: 0.9667
                                                                                              [2026-04-06 10:16:50Z] Completed

                                                                                              Workflow: MLOps CI/CD Pipeline
                                                                                              Jobs: test(1m 42s) train(8m 18s) deploy(3m 05s) - ALL PASSED
                                                                                              Model deployed to: iris-endpoint-v2
                                                                                              ```

                                                                                              ## Conclusion
                                                                                              GitHub Actions with Azure ML automates the full ML pipeline - code testing, cloud training, and model deployment - triggered on every Git push, ensuring consistent, reproducible deployments.
                                                                                              
