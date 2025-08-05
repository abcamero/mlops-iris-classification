
# 🌸 Iris Classification – End-to-End MLOps Pipeline

This project implements an end-to-end MLOps pipeline for classifying the Iris dataset using logistic regression. It covers training, model tracking with MLflow, model selection & promotion, and deployment via a FastAPI endpoint.

---

## 📁 Folder Structure

```
mlops-iris-classification/
├── data/                    # Optional raw data storage
│   └── mlruns/             # Old (ignore) MLflow artifacts
├── mlruns/                 # New MLflow run artifacts & registry (linked to HTTP server)
├── notebooks/              # Optional Jupyter notebooks
├── src/
│   ├── data/
│   │   └── iris_processed.csv
│   ├── train.py            # Model training + MLflow logging
│   ├── preprocess.py       # Dataset processing
│   ├── Selection.py        # Select best model and promote to Production
│   └── predict_api.py      # FastAPI for model prediction
├── mlflow.db               # SQLite backend for MLflow
├── .gitignore
├── Dockerfile              # Optional: containerize the app
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧪 Step-by-Step Instructions

### ✅ 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ⚙️ 2. Start MLflow Server (Option B)

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

> 📌 Keep this running in a separate terminal.  
> Access MLflow UI at: http://127.0.0.1:5001

---

### 🧼 3. (Optional) Clean Deleted Experiment Error

If you accidentally deleted an experiment:

```bash
mlflow experiments restore --experiment-id <ID>
# OR manually create a new one:
mlflow experiments create --experiment-name "Iris-Experiment"
```

---

### 📊 4. Train the Model and Log to MLflow

From the project root:

```bash
python3 src/train.py
```

- This loads `iris_processed.csv` from `src/data/`
- Trains a logistic regression model
- Logs parameters, metrics (F1 score), and model artifact to MLflow

---

### 🏷️ 5. Promote Best Model to Production

```bash
python3 src/Selection.py
```

- Lists all MLflow runs with F1 score
- Allows you to select and promote the best model to the **Production** stage

---

### 🚀 6. Start FastAPI Prediction Server

```bash
uvicorn src.predict_api:app --reload
```

API will be live at: http://127.0.0.1:8000

---

## 📡 Sample Prediction Request

Use `curl` or Postman:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "sepal length (cm)": 5.1,
           "sepal width (cm)": 3.5,
           "petal length (cm)": 1.4,
           "petal width (cm)": 0.2
         }'
```

---

## 🛑 .gitignore (Recommended)

```
__pycache__/
*.pyc
.env
mlruns/
mlflow.db
src/__pycache__/
data/
```

---

## ✅ Next Steps (Optional)

- Add CI/CD pipeline for model retraining
- Containerize FastAPI with Docker
- Push model to cloud registry (S3, Azure, GCP)
- Integrate with Prometheus & Grafana for monitoring

---

## 🙌 Credits

Built by Ameya Bodhankar using Python, scikit-learn, MLflow, and FastAPI.
