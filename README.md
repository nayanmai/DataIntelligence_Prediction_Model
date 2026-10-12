# ⚡ Data Intelligence — Prediction Lab

## An interactive machine-learning workspace for exploring multiple linear regression

Upload a dataset. Train a model. Inspect its performance. Generate a prediction.

![Python](https://img.shields.io/badge/Python-Data%20Science-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🧠 About the project

**Prediction Lab** is a browser-based application for training and exploring a multiple linear regression model from a CSV file. The regression coefficients and predictions are calculated with **NumPy**, making the underlying math visible instead of hiding it inside a prebuilt regression estimator.

The app provides a Streamlit interface for preparing mixed numeric and categorical data, evaluating test-set predictions, and trying new input values.

## ✨ What you can do

- 📂 Upload a CSV dataset and preview its records.
- 🎯 Choose a **numeric target column** for regression.
- 🧹 Prepare data by removing incomplete rows and separating numeric and categorical features.
- 🔠 Convert categorical features with scikit-learn's `OneHotEncoder`.
- 🧮 Train multiple linear regression coefficients using NumPy's pseudoinverse.
- 📊 Review **MSE**, **RMSE**, actual-versus-predicted values, and residual diagnostics.
- 🧪 Enter feature values and generate a new prediction.
- 🟢 Optionally upload a cyber-themed background image in the app sidebar.

## 🧰 Requirements & technology stack

To run Prediction Lab, you need **Python** and **pip**. Install the libraries below in your Python environment:

| Library | How it is used |
| --- | --- |
| [Streamlit](https://streamlit.io/) | Interactive web app, file upload, controls, and results |
| [pandas](https://pandas.pydata.org/) | Reading and preparing tabular CSV data |
| [NumPy](https://numpy.org/) | Regression calculations, matrix operations, and predictions |
| [scikit-learn](https://scikit-learn.org/) | Train/test split and one-hot encoding |
| [Matplotlib](https://matplotlib.org/) | Model diagnostic charts |
| [Seaborn](https://seaborn.pydata.org/) | Residual density visualization |

Use the install command in [step 3](#3-install-the-libraries) below to install them together. `Requirement_.py` is a legacy Python import list, **not** a pip requirements file.

## 🚀 Run it locally

### 1. Get the project

Clone the repository, then open a terminal in the `Prediction_model` project folder (the folder containing the Streamlit app):

```bash
git clone https://github.com/guptaharssh821-star/Prediction_model.git
cd Prediction_model
```

If you downloaded the project as a ZIP, extract it and open the extracted project folder instead.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the libraries

```bash
python -m pip install streamlit pandas numpy scikit-learn seaborn matplotlib
```

### 4. Start the app

```bash
python -m streamlit run f_6_Multi_Variate_Streamlit_.py
```

Open the local address printed in the terminal, usually **[http://localhost:8501](http://localhost:8501)**.

## 📋 Using the app

1. Upload a CSV file.
2. Select the numeric column you want the model to predict.
3. Select **Initialize model** and review the evaluation metrics and charts.
4. Enter values for the input features and request a prediction.

For a meaningful regression result, use a dataset with enough clean examples and a numeric target that represents a quantity (for example, a price, score, or measurement). The app removes rows containing missing values during training.

## 📁 Project files

```text
Prediction_model/
├── f_6_Multi_Variate_Streamlit_.py   # Streamlit app and regression implementation
├── README.md                         # Project overview and setup guide
└── Requirement_.py                   # Legacy import list (not a pip requirements file)
```

## ⚠️ Notes

- This is an educational portfolio project for experimenting with regression and data workflows.
- Model quality depends on the uploaded data, selected target, and feature relationships; results are not guaranteed.
- Avoid uploading private, sensitive, or personally identifiable information.


## Project Output:

<img width="1775" height="950" alt="Screenshot 2026-10-12 105034" src="https://github.com/user-attachments/assets/2c17e92b-ceb7-4ed4-a3e7-431565c94c68" />
<img width="1705" height="917" alt="Screenshot 2026-10-12 105046" src="https://github.com/user-attachments/assets/e63318ee-ac91-4c5f-9676-9c41211b29c3" />
<img width="1678" height="912" alt="Screenshot 2026-10-12 105057" src="https://github.com/user-attachments/assets/3a3d3717-aa56-4571-9ee9-436405a86282" />
<img width="1647" height="922" alt="Screenshot 2026-10-12 105107" src="https://github.com/user-attachments/assets/4f1bde61-eba3-4b72-ae68-f845925bf5d0" />
<img width="1652" height="910" alt="Screenshot 2026-10-12 105116" src="https://github.com/user-attachments/assets/692835ea-1801-4874-9c54-69b3dd49f5cc" />
<img width="1710" height="910" alt="Screenshot 2026-10-12 105127" src="https://github.com/user-attachments/assets/1742a0bf-c6d5-4e17-a9fb-60e3a309c182" />



## 👤 Author

**GitHub:** [@Nayan Maity] ([https://github.com/nayanmai/DataIntelligence_Prediction_Model])
---

Made with Python, NumPy, and curiosity. ✨
