# FastAPI ML App

A simple machine learning API built with FastAPI and scikit-learn that serves predictions from a Logistic Regression model trained on the Iris dataset.

## Features

- FastAPI web framework
- Logistic Regression model (binary classification)
- RESTful API endpoints
- Interactive API documentation
- Health check endpoint

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fastapi_ml_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python train.py
```

2. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

- `POST /predict` - Make custom predictions
- `GET /docs` - Interactive API documentation

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4,
    "feature4": 0.2
  }'
```

## Requirements

- Python 3.13+
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- Pydantic
