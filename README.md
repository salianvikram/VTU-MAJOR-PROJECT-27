# 🌍 Predictive Indoor/Outdoor Air Quality Control System

## 📌 Project Overview

Air pollution is one of the major environmental challenges affecting human health and quality of life. The **Air Quality Index (AQI)** provides a numerical representation of air pollution levels based on the concentration of various pollutants in the atmosphere.

This project focuses on developing and comparing multiple **Machine Learning and Deep Learning models for AQI prediction and forecasting** using historical air-quality monitoring data.

The project implements and evaluates four different approaches:

* 🌳 **Random Forest**
* 🧠 **Artificial Neural Network (ANN)**
* 🔄 **Long Short-Term Memory (LSTM)**
* 🚀 **Gradient Boosting**

The models use pollutant concentrations, temporal information, historical pollutant observations, AQI history, and rolling statistical features to predict the **AQI for the next hour**.

---

## 🎯 Objectives

The main objectives of this project are:

1. To preprocess and analyze historical air-quality data.
2. To identify important pollutants affecting AQI.
3. To engineer temporal, lag, and rolling statistical features.
4. To develop machine-learning and deep-learning models for AQI prediction.
5. To predict the **next-hour AQI** using historical observations.
6. To compare Random Forest, LSTM, Gradient Boosting, and ANN models.
7. To evaluate the models using standard regression metrics.
8. To identify the most suitable model for AQI forecasting.

---

## 🗂️ Dataset

The project uses an hourly air-quality dataset containing observations from multiple monitoring stations.

### Major attributes include:

| Feature      | Description                   |
| ------------ | ----------------------------- |
| `StationId`  | Monitoring station identifier |
| `Datetime`   | Date and time of observation  |
| `PM2.5`      | Fine particulate matter       |
| `PM10`       | Particulate matter            |
| `NO`         | Nitric oxide                  |
| `NO2`        | Nitrogen dioxide              |
| `NOx`        | Nitrogen oxides               |
| `NH3`        | Ammonia                       |
| `CO`         | Carbon monoxide               |
| `SO2`        | Sulfur dioxide                |
| `O3`         | Ozone                         |
| `Benzene`    | Benzene concentration         |
| `Toluene`    | Toluene concentration         |
| `Xylene`     | Xylene concentration          |
| `AQI`        | Air Quality Index             |
| `AQI_Bucket` | AQI category                  |

The dataset contains hourly observations from multiple monitoring stations over several years.

---

# ⚙️ Methodology

The overall methodology followed in this project is:

```text
                Raw Air Quality Dataset
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
                 Datetime Processing
                         │
                         ▼
                 Feature Engineering
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Pollutants    Time Features   Historical Data
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                    Lag Features   Rolling Features  AQI History
                         │
                         └──────────────┬──────────────┘
                                        ▼
                              Next-Hour AQI Target
                                        │
                                        ▼
                              Time-Based Data Split
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                   Random Forest       LSTM      Gradient Boosting
                         │              │              │
                         └──────────────┼──────────────┘
                                        │
                                        ▼
                                      ANN
                                        │
                                        ▼
                              Model Evaluation
                                        │
                                        ▼
                            Best AQI Prediction Model
```

---

# 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Datetime Conversion

The `Datetime` column is converted into a proper datetime format.

### 2. Chronological Sorting

The observations are sorted according to:

```text
StationId → Datetime
```

This ensures that historical features are generated in chronological order.

### 3. Numerical Conversion

Pollutant and AQI values are converted into numerical format, with invalid values handled appropriately.

### 4. Missing Value Handling

Rows containing missing values in the required modelling features are removed before model training.

### 5. Infinite Value Handling

Infinite values are replaced with missing values and subsequently removed.

---

# 🛠️ Feature Engineering

Feature engineering is an important part of this project.

## 1. Pollutant Features

The models use available pollutant concentrations including:

```text
PM2.5
PM10
NO
NO2
NOx
NH3
CO
SO2
O3
Benzene
Toluene
Xylene
```

## 2. Time Features

The following temporal features are extracted from `Datetime`:

```text
Hour
Day
Month
Day of Week
Day of Year
Week of Year
Year
Weekend Indicator
```

## 3. Cyclic Features

Cyclic encoding is applied to capture periodic patterns:

```text
Hour_Sin
Hour_Cos
Month_Sin
Month_Cos
DayOfWeek_Sin
DayOfWeek_Cos
```

This helps the models understand that, for example, hour 23 and hour 0 are close to each other in time.

## 4. Lag Features

Historical observations are incorporated using lag features:

```text
1 hour
3 hours
6 hours
12 hours
24 hours
```

For example:

```text
PM2.5_lag_1
PM2.5_lag_3
PM2.5_lag_6
PM2.5_lag_12
PM2.5_lag_24
```

Historical AQI values are also incorporated using the same lag periods.

## 5. Rolling Features

Rolling averages are calculated over:

```text
3 observations
6 observations
24 observations
```

These features capture recent pollution trends and smooth short-term fluctuations.

---

# 🎯 Target Variable

The target variable is:

```text
AQI_Next_Hour
```

The model uses information available at time `t` to predict AQI at time `t + 1`.

```text
Current & Historical Information
             │
             ▼
          ML Model
             │
             ▼
       Next-Hour AQI
```

This makes the project a **time-series forecasting problem** rather than simply estimating AQI from measurements at the same timestamp.

---

# 📊 Train-Test Strategy

A chronological split is used instead of random splitting.

```text
2015 ─────────────── 2019 │ 2020
        TRAINING          │ TESTING
```

### Training Dataset

**2015–2019**

### Testing Dataset

**2020**

This approach prevents future observations from being randomly mixed into the training dataset and provides a more realistic evaluation of forecasting performance.

---

# 🤖 Machine Learning Models

## 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It is used as a strong tree-based baseline for AQI forecasting.

### Advantages

* Handles nonlinear relationships.
* Works well with many features.
* Robust to feature interactions.
* Provides feature importance.

Notebook:

`RFMODEL.ipynb`

---

# 2. Long Short-Term Memory (LSTM)

LSTM is a recurrent neural-network architecture designed for sequential and time-dependent data.

The LSTM model uses historical observations to learn temporal dependencies in air-quality patterns.

A sequence of historical observations is provided to the network to forecast the next-hour AQI.

### Advantages

* Suitable for time-series data.
* Captures temporal dependencies.
* Can learn complex sequential patterns.

Notebook:

`LSTM_MODEL.ipynb`

---

# 3. Gradient Boosting

Gradient Boosting is an ensemble learning technique where decision trees are built sequentially, with each stage attempting to improve the errors of previous stages.

For this large dataset, **Histogram-based Gradient Boosting** is used for improved computational efficiency.

### Advantages

* Strong nonlinear modelling capability.
* Effective with structured/tabular data.
* Can capture complex feature interactions.
* Efficient for large datasets.

Notebook:

`GB_MODEL.ipynb`

---

# 4. Artificial Neural Network (ANN)

The ANN model uses fully connected neural-network layers to learn nonlinear relationships between the engineered air-quality features and future AQI.

The architecture consists of multiple dense layers with ReLU activation and dropout regularization.

### Architecture

```text
Input Features
      │
      ▼
Dense Layer (128)
      │
    Dropout
      │
      ▼
Dense Layer (64)
      │
    Dropout
      │
      ▼
Dense Layer (32)
      │
      ▼
Output Layer (1)
      │
      ▼
Next-Hour AQI
```

Notebook:

`ANN_MODEL.ipynb`

---

# 📏 Evaluation Metrics

The models are evaluated using four regression metrics.

## Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted AQI.

```text
MAE = average(|Actual - Predicted|)
```

Lower MAE indicates better performance.

---

## Mean Squared Error (MSE)

Measures the average squared prediction error.

```text
MSE = average((Actual - Predicted)²)
```

Lower MSE indicates better performance.

---

## Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

```text
RMSE = √MSE
```

Lower RMSE indicates better performance.

---

## R² Score

R² measures the proportion of variance in AQI explained by the model.

```text
R² = 1 - SS_res / SS_tot
```

A value closer to `1` indicates stronger predictive performance.

---

# 📈 Model Comparison

The final performance of the four models can be summarized as:

| Model             | MAE | RMSE | R² |
| ----------------- | --: | ---: | -: |
| Random Forest     |   — |    — |  — |
| LSTM              |   — |    — |  — |
| Gradient Boosting |   — |    — |  — |
| ANN               |   — |    — |  — |

> **Note:** Replace the values above with the final test-set results obtained from the respective notebooks.

The model with the **lowest MAE/RMSE and highest R²** can be considered the strongest performer under the experimental setup.

---

# 📂 Repository Structure

```text
VTU-MAJOR-PROJECT-27/
│
├── ANN_MODEL.ipynb
│
├── GB_MODEL.ipynb
│
├── LSTM_MODEL.ipynb
│
├── RFMODEL.ipynb
│
├── Dataset Used/
│
└── README.md
```

---

# 💻 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Joblib

### Machine Learning

* Random Forest Regressor
* Histogram Gradient Boosting

### Deep Learning

* Artificial Neural Network
* Long Short-Term Memory (LSTM)

### Development Environment

* Google Colab
* Jupyter Notebook
* GitHub

---

# 🚀 How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/salianvikram/VTU-MAJOR-PROJECT-27.git
```

## Step 2 — Open the Repository

Navigate into the project directory:

```bash
cd VTU-MAJOR-PROJECT-27
```

## Step 3 — Open the Notebooks

The repository contains separate notebooks for each model:

```text
RFMODEL.ipynb
LSTM_MODEL.ipynb
GB_MODEL.ipynb
ANN_MODEL.ipynb
```

They can be opened using **Google Colab or Jupyter Notebook**.

## Step 4 — Dataset

Ensure that the required dataset is available in the expected location before executing the notebooks.

## Step 5 — Execute the Notebook

Run the notebook cells sequentially to:

1. Load the dataset.
2. Preprocess the data.
3. Generate features.
4. Train the model.
5. Evaluate the model.
6. Visualize predictions.

---

# 📌 Project Highlights

* Uses **real-world hourly air-quality data**.
* Supports data from multiple monitoring stations.
* Uses multiple pollutant parameters.
* Incorporates temporal patterns.
* Uses historical lag information.
* Uses rolling statistical features.
* Uses a chronological train-test split.
* Compares both machine-learning and deep-learning approaches.
* Performs next-hour AQI forecasting.
* Provides multiple evaluation metrics.
* Includes visualization of actual versus predicted AQI.

---

# 🔮 Future Enhancements

The project can be extended in several ways:

* Develop a real-time AQI prediction system.
* Integrate live air-quality sensor/API data.
* Develop a web-based AQI monitoring dashboard.
* Implement multi-step AQI forecasting.
* Perform hyperparameter optimization.
* Investigate advanced models such as GRU, XGBoost, LightGBM, and Transformer architectures.
* Deploy the best-performing model as a web or mobile application.
* Add interactive AQI visualizations.
* Incorporate weather parameters such as temperature, humidity, wind speed, and atmospheric pressure.
* Develop separate indoor and outdoor AQI prediction pipelines.

---

# 🏆 Expected Outcome

The primary outcome of the project is a comparative analysis of different machine-learning and deep-learning techniques for predicting future AQI.

The comparison helps determine how effectively different modelling approaches capture:

* Pollutant interactions
* Temporal patterns
* Historical AQI trends
* Short-term pollution fluctuations

The final selected model can subsequently be used as the foundation for an AQI forecasting and simulation system.

---

# 👨‍💻 Project

**VTU Major Project**

**Project Title:**
**Indoor/Outdoor Air Quality Index Prediction and Simulation**

**Repository:**
https://github.com/salianvikram/VTU-MAJOR-PROJECT-27

---

# 📜 License

This project is developed for academic and educational purposes as part of a VTU Major Project.
