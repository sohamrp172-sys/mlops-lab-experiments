# Experiment 7: Managing Data and Models using DVC with Cloud Remote

**Name:** Soham | **Subject:** MLOps Lab | **Date:** April 2026

## Aim
To study data and model versioning using DVC (Data Version Control) with Azure Blob Storage as a cloud remote backend.

## Objective
- Understand why Git cannot handle large ML files and how DVC solves this
- Learn core DVC workflow: dvc init, dvc add, dvc push, dvc pull
- Set up Azure Blob Storage as a DVC remote
- Build a DVC pipeline with prepare and train stages
- Compare two experiment versions using dvc metrics diff and dvc params diff

## Theory

The Problem: Git is designed for small text files. ML datasets (GBs-TBs) and model binaries (.pkl, .h5, .pt) exceed GitHub's 100MB file limit and bloat repositories.

DVC Solution: Large files go to cloud remote (Azure Blob / GCS / S3). Small .dvc metadata files (containing MD5 hash) go to Git. Running \"dvc pull\" restores all large files using the hash.

DVC Pipeline (dvc.yaml) defines stages with:
- cmd: command to run
- deps: input dependencies (triggers rerun if changed)
- outs: output files (cached by DVC)
- metrics: files tracked as metrics (not cached)

## Code: src/create_dataset.py

    import pandas as pd
        from sklearn.datasets import load_iris
            import os

                iris = load_iris()
                    df = pd.DataFrame(iris.data, columns=iris.feature_names)
                        df['target']  = iris.target
                            df['species'] = df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
                                os.makedirs(\"data/raw\", exist_ok=True)
                                    df.to_csv(\"data/raw/iris.csv\", index=False)

                                    ## Code: src/prepare.py

                                        import pandas as pd
                                            from sklearn.preprocessing import StandardScaler
                                                import os

                                                    df = pd.read_csv(\"data/raw/iris.csv\")
                                                        df['petal_area']  = df['petal length (cm)'] * df['petal width (cm)']
                                                            df['sepal_area']  = df['sepal length (cm)'] * df['sepal width (cm)']
                                                                scaler = StandardScaler()
                                                                    feature_cols = [c for c in df.columns if c not in ['target', 'species']]
                                                                        df[feature_cols] = scaler.fit_transform(df[feature_cols])
                                                                            os.makedirs(\"data/processed\", exist_ok=True)
                                                                                df.to_csv(\"data/processed/features.csv\", index=False)

                                                                                ## Code: src/train.py

                                                                                    import pandas as pd, json, os, pickle, yaml
                                                                                        from sklearn.model_selection import train_test_split
                                                                                            from sklearn.ensemble import GradientBoostingClassifier
                                                                                                from sklearn.metrics import accuracy_score

                                                                                                    with open(\"params.yaml\") as f:
                                                                                                            params = yaml.safe_load(f)
                                                                                                            
                                                                                                                df = pd.read_csv(\"data/processed/features.csv\")
                                                                                                                    X, y = df.drop(['target', 'species'], axis=1), df['target']
                                                                                                                        X_train, X_test, y_train, y_test = train_test_split(
                                                                                                                                X, y, test_size=params['test_size'], random_state=params['seed']
                                                                                                                                    )
                                                                                                                                        model = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
                                                                                                                                            model.fit(X_train, y_train)
                                                                                                                                                acc = accuracy_score(y_test, model.predict(X_test))
                                                                                                                                                    os.makedirs(\"metrics\", exist_ok=True)
                                                                                                                                                        with open(\"metrics/scores.json\", \"w\") as f: json.dump({\"accuracy\": acc}, f)
                                                                                                                                                        
                                                                                                                                                        ## Code: params.yaml
                                                                                                                                                        
                                                                                                                                                            test_size: 0.2
                                                                                                                                                                seed: 42
                                                                                                                                                                    n_estimators: 100
                                                                                                                                                                        learning_rate: 0.1
                                                                                                                                                                        
                                                                                                                                                                        ## Code: dvc.yaml
                                                                                                                                                                        
                                                                                                                                                                            stages:
                                                                                                                                                                                  prepare:
                                                                                                                                                                                          cmd: python src/prepare.py
                                                                                                                                                                                                  deps:
                                                                                                                                                                                                            - data/raw/iris.csv
                                                                                                                                                                                                                      - src/prepare.py
                                                                                                                                                                                                                              outs:
                                                                                                                                                                                                                                        - data/processed/features.csv
                                                                                                                                                                                                                                              train:
                                                                                                                                                                                                                                                      cmd: python src/train.py
                                                                                                                                                                                                                                                              deps:
                                                                                                                                                                                                                                                                        - data/processed/features.csv
                                                                                                                                                                                                                                                                                  - src/train.py
                                                                                                                                                                                                                                                                                            - params.yaml
                                                                                                                                                                                                                                                                                                    metrics:
                                                                                                                                                                                                                                                                                                              - metrics/scores.json
                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                              ## Commands
                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                  pip install dvc dvc-azure
                                                                                                                                                                                                                                                                                                                      dvc init
                                                                                                                                                                                                                                                                                                                          python src/create_dataset.py
                                                                                                                                                                                                                                                                                                                              dvc add data/raw/iris.csv
                                                                                                                                                                                                                                                                                                                                  dvc remote add -d azure_remote azure://mlops-dvc/data
                                                                                                                                                                                                                                                                                                                                      dvc push
                                                                                                                                                                                                                                                                                                                                          dvc repro
                                                                                                                                                                                                                                                                                                                                              dvc metrics diff
                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                              ## Output
                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                              ### dvc add
                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                  100%|#####| 1/1 [00:00, 20.35file/s]
                                                                                                                                                                                                                                                                                                                                                      Adding 'data/raw/iris.csv' to '.gitignore'.
                                                                                                                                                                                                                                                                                                                                                          Adding 'data/raw/iris.csv.dvc' to Git.
                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                          ### dvc repro
                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                              Running stage 'prepare':
                                                                                                                                                                                                                                                                                                                                                                  > python src/prepare.py
                                                                                                                                                                                                                                                                                                                                                                      Running stage 'train':
                                                                                                                                                                                                                                                                                                                                                                          > python src/train.py
                                                                                                                                                                                                                                                                                                                                                                              Updating lock file 'dvc.lock'
                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                              ### dvc metrics diff
                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                  Path                Metric    Old       New       Change
                                                                                                                                                                                                                                                                                                                                                                                      metrics/scores.json accuracy  0.93333   0.96667   0.03334
                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                      ## Conclusion
                                                                                                                                                                                                                                                                                                                                                                                      DVC solves the large file versioning problem for ML. By separating content from metadata, full experiment reproducibility is achieved. Pipeline stage caching skips unchanged stages, making DVC indispensable for production MLOps.
                                                                                                                                                                                                                                                                                                                                                                                      
