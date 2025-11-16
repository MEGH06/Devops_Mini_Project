# Diabetes Prediction with PyCaret - Complete Guide

## 📋 What is PyCaret?

**PyCaret** is a low-code machine learning library that automates the entire ML workflow:

- **Data Preprocessing**: Handles encoding, scaling, missing values automatically
- **Model Comparison**: Trains 15+ models in one line and ranks them
- **Hyperparameter Tuning**: Optimizes model parameters automatically
- **Model Evaluation**: Generates comprehensive visualizations
- **Deployment Ready**: Easy model saving/loading

Think of it as **AutoML made simple** - you get production-ready models with minimal code!

---

## 🚀 How to Run This Project

### Method 1: Local Setup (Without Docker)

#### Step 1: Install Dependencies

```bash
pip install pandas pycaret scikit-learn
```

#### Step 2: Prepare Your Data

Make sure `diabetes_dataset.csv` is in the same directory as your Python file.

#### Step 3: Run the Script

```bash
python devops_miniproject_pycaret.py
```

#### What to Expect:

1. **Setup Phase** (~30 seconds): PyCaret analyzes your data and prepares it
2. **Model Comparison** (~2-5 minutes): Trains 15+ models and shows a comparison table
3. **Model Tuning** (~1-3 minutes): Optimizes the best model
4. **Evaluation**: Shows various plots and metrics
5. **Model Saved**: Creates `diabetes_prediction_model.pkl`

---

### Method 2: Docker Setup (Recommended for Deployment)

#### Project Structure:

```
diabetes-prediction/
├── devops_miniproject_pycaret.py
├── diabetes_dataset.csv
├── Dockerfile
├── requirements.txt
└── README.md
```

#### Step 1: Create `requirements.txt`

```txt
pandas==2.0.3
pycaret==3.0.4
scikit-learn==1.3.0
numpy==1.24.3
```

#### Step 2: Create `Dockerfile`

```dockerfile
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY devops_miniproject_pycaret.py .
COPY diabetes_dataset.csv .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "devops_miniproject_pycaret.py"]
```

#### Step 3: Build Docker Image

```bash
docker build -t diabetes-prediction:v1 .
```

#### Step 4: Run Docker Container

```bash
docker run --name diabetes-ml diabetes-prediction:v1
```

#### Step 5: Copy Model from Container (Optional)

```bash
# Copy the trained model from container to your local machine
docker cp diabetes-ml:/app/diabetes_prediction_model.pkl .
```

---

## 📊 Expected Output Explanation

### 1. **Setup Output**

```
  Description                Value
  session_id                 42
  Target                     diagnosed_diabetes
  Target Type                Binary
  Original Data              (10000, 15)
  Transformed Data           (10000, 20)  # After encoding
  Train Set                  (8000, 20)
  Test Set                   (2000, 20)
```

### 2. **Model Comparison Table**

You'll see a table ranking all models:

```
Model                    Accuracy  AUC    Recall  Prec.   F1      Kappa   TT(Sec)
Random Forest            0.9245    0.9823 0.9156  0.9334  0.9244  0.8489  2.341
Extra Trees              0.9210    0.9801 0.9123  0.9298  0.9209  0.8419  1.876
Gradient Boosting        0.9189    0.9787 0.9089  0.9289  0.9188  0.8377  3.234
...
```

### 3. **Tuning Results**

Shows improvement after hyperparameter optimization:

```
Before Tuning: Accuracy = 0.9245
After Tuning:  Accuracy = 0.9312 ✓
```

### 4. **Visualizations Generated**

- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve
- Feature Importance Plot
- Class Prediction Error

### 5. **Final Model File**

- `diabetes_prediction_model.pkl` - Your trained model ready for deployment!

---

## 🔧 Key PyCaret Functions Explained

### `setup()`

**Purpose**: Initialize PyCaret environment and preprocess data

```python
clf = setup(data=df, target='diagnosed_diabetes')
```

**What it does**:

- Splits data into train/test
- Encodes categorical variables
- Scales numerical features
- Handles missing values

### `compare_models()`

**Purpose**: Train and compare multiple models

```python
best_models = compare_models(n_select=3)
```

**Models tested**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Logistic Regression, KNN, Naive Bayes, SVM, Decision Tree, and more!

### `tune_model()`

**Purpose**: Optimize hyperparameters

```python
tuned_model = tune_model(best_model, n_iter=10)
```

**Uses**: Random/Grid search to find best parameters

### `evaluate_model()`

**Purpose**: Generate all evaluation plots

```python
evaluate_model(tuned_model)
```

### `save_model()` / `load_model()`

**Purpose**: Save and load trained models

```python
save_model(tuned_model, 'my_model')
loaded = load_model('my_model')
```

---

## 🎯 Advantages of PyCaret Over Your Original Code

| Feature               | Original Code     | PyCaret                  |
| --------------------- | ----------------- | ------------------------ |
| Models Tested         | 1 (Random Forest) | 15+ models automatically |
| Hyperparameter Tuning | Manual            | Automatic                |
| Model Comparison      | None              | Built-in ranking         |
| Visualizations        | Manual coding     | One-line plots           |
| Code Length           | ~50 lines         | ~20 lines                |
| Preprocessing         | Manual            | Automatic                |
| Time to Production    | Days              | Hours                    |

---

## 🐳 Docker Best Practices

### Multi-stage Build (Advanced - Smaller Image)

```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY devops_miniproject_pycaret.py .
COPY diabetes_dataset.csv .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "devops_miniproject_pycaret.py"]
```

### Useful Docker Commands

```bash
# Build image
docker build -t diabetes-prediction:v1 .

# Run container
docker run diabetes-prediction:v1

# Run interactively (for debugging)
docker run -it diabetes-prediction:v1 bash

# Check logs
docker logs diabetes-ml

# Remove container
docker rm diabetes-ml

# Remove image
docker rmi diabetes-prediction:v1

# Push to Docker Hub
docker tag diabetes-prediction:v1 yourusername/diabetes-prediction:v1
docker push yourusername/diabetes-prediction:v1
```

---

## 📈 Next Steps for Your Mini Project

1. **API Development**: Create a Flask/FastAPI endpoint to serve predictions
2. **Web Interface**: Build a simple HTML form for user input
3. **CI/CD Pipeline**: Set up GitHub Actions to auto-build Docker images
4. **Cloud Deployment**: Deploy to AWS/Azure/GCP
5. **Monitoring**: Add logging and performance tracking

---

## 🆘 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'pycaret'`

- **Fix**: `pip install pycaret`

**Issue**: Docker build fails with memory error

- **Fix**: Increase Docker memory limit in Docker Desktop settings

**Issue**: PyCaret setup takes too long

- **Fix**: Reduce dataset size or set `html=False` in setup

**Issue**: Plots not showing

- **Fix**: Add `save=True` parameter to `plot_model()` function

---

## 📚 Resources

- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)
- [PyCaret GitHub](https://github.com/pycaret/pycaret)
- [Docker Documentation](https://docs.docker.com/)

---

**Good luck with your DevOps mini project! 🚀**
