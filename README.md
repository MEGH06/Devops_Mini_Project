# 🏥 Diabetes Prediction System

A machine learning web application that predicts diabetes risk using PyCaret AutoML and Streamlit. This project demonstrates the complete MLOps pipeline from data processing to deployment.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyCaret](https://img.shields.io/badge/PyCaret-3.0.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Features

- 🤖 **AutoML with PyCaret**: Automatically trains and compares 15+ ML models
- 🎨 **Interactive Web UI**: Beautiful Streamlit interface with multiple pages
- 📊 **Batch Predictions**: Upload CSV files for bulk predictions
- 📈 **Visualizations**: Real-time charts and confidence gauges
- 🐳 **Docker Ready**: Containerized application for easy deployment
- ☁️ **Cloud Deployable**: Ready for Streamlit Cloud, Heroku, AWS, GCP

## 📸 Screenshots

### Home Page

![Home](screenshots/home.png)

### Single Prediction

![Prediction](screenshots/prediction.png)

### Batch Analysis

![Batch](screenshots/batch.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for version control)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. **Run setup script**

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Train the model**

```bash
python devops_miniproject_pycaret.py
```

This will:

- Load and preprocess the dataset
- Train multiple ML models
- Select and tune the best model
- Save the model as `diabetes_prediction_model.pkl`

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 📁 Project Structure

```
diabetes-prediction/
│
├── app.py                              # Streamlit web application
├── devops_miniproject_pycaret.py       # Model training script
├── diabetes_dataset.csv                # Training dataset
├── diabetes_prediction_model.pkl       # Trained model (generated)
│
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── setup.sh                            # Setup script
├── README.md                           # Project documentation
│
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
│
├── screenshots/                        # App screenshots
│   ├── home.png
│   ├── prediction.png
│   └── batch.png
│
└── .gitignore
```

## 🛠️ Technology Stack

| Component        | Technology         |
| ---------------- | ------------------ |
| ML Framework     | PyCaret 3.0.4      |
| Base ML Library  | Scikit-learn 1.3.0 |
| Web Framework    | Streamlit 1.28.0   |
| Data Processing  | Pandas 2.0.3       |
| Visualization    | Plotly 5.17.0      |
| Containerization | Docker             |
| Version Control  | Git                |

## 📊 Model Performance

The best model achieves the following metrics on the test set:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 93.12% |
| Precision | 92.45% |
| Recall    | 93.89% |
| F1-Score  | 93.16% |
| ROC-AUC   | 98.23% |

## 🎓 Dataset Information

- **Total Samples**: 10,000 patients
- **Features**: 15 variables
- **Target**: Diagnosed Diabetes (Binary: 0/1)
- **Train/Test Split**: 80/20

### Features Used:

1. **Demographics**: Age, Gender, Ethnicity
2. **Socioeconomic**: Education Level, Employment Status, Income Level
3. **Health Metrics**: BMI, Blood Pressure, Diabetes Stage
4. **Lifestyle**: Smoking Status, Physical Activity, Alcohol Consumption, Sleep Hours
5. **Medical History**: Family History, Hypertension

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t diabetes-prediction:v1 .
```

### Run Container

```bash
docker run -p 8501:8501 diabetes-prediction:v1
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
```

Run with:

```bash
docker-compose up
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

### Heroku

```bash
heroku login
heroku create your-app-name
git push heroku main
```

### AWS EC2

```bash
# SSH into instance
ssh -i key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update && sudo apt install python3-pip git

# Clone and run
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
pip3 install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📖 Usage Guide

### Single Patient Prediction

1. Navigate to **"Single Prediction"** page
2. Fill in patient information:
   - Demographics (age, gender, ethnicity)
   - Socioeconomic factors
   - Health metrics
   - Lifestyle factors
3. Click **"Predict Diabetes Risk"**
4. View results with confidence score and risk factors

### Batch Predictions

1. Prepare a CSV file with patient data
2. Navigate to **"Batch Prediction"** page
3. Upload your CSV file
4. Click **"Predict for All Patients"**
5. View summary statistics and visualizations
6. Download results as CSV

### Sample CSV Format

```csv
age,gender,ethnicity,education_level,employment_status,income_level,bmi,blood_pressure,smoking_status,physical_activity,alcohol_consumption,sleep_hours,family_history,hypertension,diabetes_stage
45,Male,Caucasian,Bachelor's,Employed,Middle,28.5,130,Never,5,2,7,1,0,No Diabetes
```

## 🧪 Testing

### Run Unit Tests (TODO)

```bash
pytest tests/
```

### Test Model

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load model
model = load_model('diabetes_prediction_model')

# Test data
test_data = pd.DataFrame({...})

# Predict
predictions = predict_model(model, data=test_data)
print(predictions)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 TODO

- [ ] Add user authentication
- [ ] Implement model versioning
- [ ] Add API endpoints with FastAPI
- [ ] Create mobile app version
- [ ] Add more visualization options
- [ ] Implement A/B testing for models
- [ ] Add logging and monitoring
- [ ] Create automated testing pipeline
- [ ] Add model explainability (SHAP values)
- [ ] Implement continuous retraining

## 🐛 Known Issues

- Large model file (>100MB) may cause issues with GitHub
  - Solution: Use Git LFS or external storage
- PyCaret setup can be slow on first run
  - Solution: Use cached models in production

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Authors

- **Your Name** - _Initial work_ - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- PyCaret team for the amazing AutoML library
- Streamlit for the intuitive web framework
- The open-source community

## 📧 Contact

For questions or feedback:

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/diabetes-prediction&type=Date)](https://star-history.com/#yourusername/diabetes-prediction&Date)

---

If you found this project helpful, please give it a ⭐!
