# Heart Disease Prediction Model

This is a Machine Learning project that predicts whether a person has heart disease based on different medical features.

In this project, I used multiple Machine Learning algorithms and compared their performance. After comparing the models, I selected **K-Nearest Neighbors (KNN)** because it gave the highest accuracy.

I also built and hosted a **Streamlit web application** where users can enter patient information and get a heart disease prediction.

## Live Demo

The project is hosted using Streamlit:

[Open the Streamlit App](YOUR_STREAMLIT_APP_LINK_HERE)

## Dataset

The dataset contains information about patients such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

The target variable is `HeartDisease`.

- `0` = No Heart Disease
- `1` = Heart Disease

## Machine Learning Models Used

I trained and compared the following models:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

After comparing the models, KNN performed the best.

## Model Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.50% |
| KNN | **88.59%** |
| Gaussian Naive Bayes | 86.96% |
| Decision Tree | 75.00% |
| SVM | 86.41% |

Since KNN gave the highest accuracy of **88.59%**, I selected it as the final model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

The project follows these steps:

1. Load the dataset
2. Perform data cleaning
3. Perform Exploratory Data Analysis (EDA)
4. Encode categorical features
5. Split the data into training and testing sets
6. Scale the features using StandardScaler
7. Train different Machine Learning models
8. Compare the model performances
9. Select KNN as the best model
10. Save the trained model
11. Build a Streamlit application
12. Deploy the application using Streamlit

## Files in the Project

- `Heart_Disease_Prediction.ipynb` - Jupyter Notebook containing the complete ML process
- `heart.csv` - Dataset
- `app.py` - Streamlit application
- `KNN_heart.pkl` - Saved KNN model
- `scaler.pkl` - Saved scaler
- `columns.pkl` - Saved column information
- `requirements.txt` - Required Python libraries

## How to Run

First, clone the repository:

```bash
git clone https://github.com/ankan7703/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model
pip install -r requirements.txt
streamlit run app.py
## Result

Among all the tested models, **KNN performed the best with an accuracy of 88.59%**. Therefore, KNN was selected as the final model for the heart disease prediction application.

The application is hosted using Streamlit and can be accessed through the Live Demo link above.

## Disclaimer

This project is made for learning and educational purposes only. It should not be used for actual medical diagnosis.

## Author

**Ankan Paul**

GitHub: [ankan7703](https://github.com/ankan7703)
