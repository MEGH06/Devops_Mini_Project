# 🚀 Streamlit Deployment Guide - Diabetes Prediction App

## 📁 Complete Project Structure

```
diabetes-prediction-app/
├── app.py                              # Streamlit app
├── devops_miniproject_pycaret.py       # Model training script
├── diabetes_dataset.csv                # Your dataset
├── diabetes_prediction_model.pkl       # Trained model (generated)
├── requirements.txt                    # Dependencies
├── Dockerfile                          # For Docker deployment
├── .streamlit/
│   └── config.toml                     # Streamlit config (optional)
└── README.md
```

---

## 🎯 Step-by-Step Deployment Process

### Step 1: Train Your Model First

Before running the Streamlit app, you **must** train the model:

```bash
python devops_miniproject_pycaret.py
```

This will create `diabetes_prediction_model.pkl` file which the Streamlit app needs.

---

### Step 2: Test Locally

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud (Recommended & Free!)

### Method 1: Deploy via GitHub

#### Step 1: Push to GitHub

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Diabetes prediction app"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/diabetes-prediction.git

# Push
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path:** `app.py`
   - **Python version:** 3.9
6. Click "Deploy"

#### ⚠️ Important: Model File Handling

Since GitHub has file size limits, you have two options:

**Option A: Upload model via Streamlit Secrets (Recommended)**

The model file might be too large for GitHub. Instead:

1. Upload `diabetes_prediction_model.pkl` to Google Drive or Dropbox
2. Get a direct download link
3. Modify your `app.py` to download the model on startup:

```python
import streamlit as st
import os
import requests

@st.cache_resource
def download_model():
    model_url = "YOUR_DIRECT_DOWNLOAD_LINK"
    model_path = "diabetes_prediction_model.pkl"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)

    return load_model('diabetes_prediction_model')
```

**Option B: Use Git LFS (Large File Storage)**

```bash
# Install Git LFS
git lfs install

# Track .pkl files
git lfs track "*.pkl"

# Add and commit
git add .gitattributes
git add diabetes_prediction_model.pkl
git commit -m "Add model file with Git LFS"
git push
```

---

### Method 2: Deploy with Docker on Streamlit Cloud

#### Create `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

#### Updated Dockerfile for Streamlit

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run

```bash
# Build image
docker build -t diabetes-streamlit:v1 .

# Run container
docker run -p 8501:8501 diabetes-streamlit:v1
```

Access at `http://localhost:8501`

---

## ☁️ Alternative Deployment Options

### Option 1: Heroku

#### Create `Procfile`

```
web: sh setup.sh && streamlit run app.py
```

#### Create `setup.sh`

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

#### Deploy

```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

### Option 2: AWS EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip

# Clone your repo
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

# Install requirements
pip3 install -r requirements.txt

# Run with nohup (keeps running after logout)
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Or use tmux/screen for persistent session
```

Don't forget to:

- Open port 8501 in security group
- Use elastic IP for persistent address

---

### Option 3: Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/diabetes-app

# Deploy to Cloud Run
gcloud run deploy diabetes-app \
  --image gcr.io/PROJECT_ID/diabetes-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 📊 App Features Overview

Your Streamlit app includes:

### 🏠 **Home Page**

- App overview and description
- Model performance metrics visualization
- Instructions for users

### 🔮 **Single Prediction**

- Interactive form for patient data input
- Real-time prediction with confidence score
- Visual gauge for confidence level
- Risk factors analysis
- Health recommendations

### 📊 **Batch Prediction**

- CSV file upload
- Bulk predictions for multiple patients
- Summary statistics and visualizations
- Downloadable results

### 📈 **Model Info**

- Model details and specifications
- Performance metrics table
- Feature importance visualization
- Disclaimers

---

## 🎨 Customization Tips

### Change Theme Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"        # Red
backgroundColor = "#0E1117"      # Dark background
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

### Add Custom Logo

```python
from PIL import Image

# At the top of app.py
logo = Image.open("logo.png")
st.sidebar.image(logo, width=200)
```

### Add Authentication

```python
import streamlit_authenticator as stauth

# Simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == "your_password":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password",
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password",
                     on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app code here
```

---

## 🔧 Troubleshooting

### Issue: Model file not found

**Solution:** Make sure you run `devops_miniproject_pycaret.py` first to generate the model

### Issue: Streamlit app won't start

**Solution:**

```bash
# Clear Streamlit cache
streamlit cache clear

# Check if port is in use
lsof -i :8501

# Kill process on port
kill -9 <PID>
```

### Issue: Memory error on Streamlit Cloud

**Solution:**

- Reduce model size by using feature selection
- Use lighter model (e.g., Logistic Regression instead of ensemble)
- Implement model quantization

### Issue: Slow predictions

**Solution:**

```python
# Add caching to model loading
@st.cache_resource
def load_trained_model():
    return load_model('diabetes_prediction_model')
```

---

## 📱 Testing Checklist

Before deploying, test:

- ✅ Single prediction works
- ✅ Batch prediction accepts CSV
- ✅ All visualizations render correctly
- ✅ Download button works
- ✅ Forms validate input correctly
- ✅ Error handling for bad inputs
- ✅ Mobile responsive design
- ✅ All pages navigate correctly

---

## 🎓 Sample CSV for Testing Batch Prediction

Create `sample_patients.csv`:

```csv
age,gender,ethnicity,education_level,employment_status,income_level,bmi,blood_pressure,smoking_status,physical_activity,alcohol_consumption,sleep_hours,family_history,hypertension,diabetes_stage
45,Male,Caucasian,Bachelor's,Employed,Middle,28.5,130,Never,5,2,7,1,0,No Diabetes
52,Female,Asian,Master's,Self-Employed,High,24.2,118,Former,8,1,8,0,0,No Diabetes
38,Male,Hispanic,High School,Unemployed,Low,32.1,145,Current,2,5,6,1,1,Prediabetes
```

---

## 🚀 Performance Optimization

### 1. Enable Caching

```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return load_model('model.pkl')
```

### 2. Lazy Loading

```python
# Load model only when needed
if 'model' not in st.session_state:
    st.session_state.model = load_trained_model()
```

### 3. Optimize Images

- Use compressed images
- Lazy load images
- Use appropriate image formats (WebP)

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)

---

## 🎉 Next Steps

1. ✅ Train your model
2. ✅ Test locally
3. ✅ Push to GitHub
4. ✅ Deploy to Streamlit Cloud
5. ✅ Share with the world!

**Your app will be live at:** `https://your-app-name.streamlit.app`

Good luck with your deployment! 🚀
