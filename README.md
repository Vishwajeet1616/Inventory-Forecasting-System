# 📦 DemandSense AI: Inventory Forecasting System

A professional Machine Learning web application designed to optimize supply chain management by predicting product demand. This project features a comparative analysis of multiple architectures, ultimately deploying a **Random Forest Regressor** for its balance of accuracy and computational efficiency.



## 🚀 Features
* **Multi-Model Benchmarking:** Evaluated ARIMA, LSTM, and Random Forest for time-series accuracy.
* **Time-Series Forecasting:** Converts raw transaction data into weekly sales aggregates.
* **Feature Engineering:** Implements a 4-week rolling lag system to capture seasonal trends.
* **Interactive Dashboard:** A modern, responsive Flask-based UI for real-time predictions.
* **Dynamic Visualization:** Generates automated Matplotlib charts for historical vs. predicted sales.

## 🧠 Model Evaluation & Selection
Before building the final application, a rigorous benchmarking process was conducted to find the best-performing model:

| Model | Approach | Strength | Result |
| :--- | :--- | :--- | :--- |
| **ARIMA** | Statistical | Great for linear trends | Struggled with retail volatility |
| **LSTM (RNN)** | Deep Learning | Captures long-term sequences | Overfitted on small-batch SKU data |
| **Random Forest** | **Ensemble ML** | **Robust to outliers & non-linear** | **Best performance (Selected)** |

> **Selection Note:** Random Forest was chosen because it handled the variance in individual stock codes significantly better than statistical models, while requiring less data and training time than the LSTM.



## 🛠️ Tech Stack
* **Backend:** Python (Flask)
* **Machine Learning:** Scikit-Learn (Random Forest), Statsmodels (ARIMA), TensorFlow (LSTM)
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML5, CSS3 (Professional Sidebar Layout)
* **Visualization:** Matplotlib

## 📊 Dataset
The project uses the **Online Retail II** dataset, containing all transactions occurring between 01/12/2009 and 09/12/2011 for a UK-based online retail store.

* **Source:** [Kaggle - Online Retail II Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-listing)
* **Size:** ~90MB (CSV)
* **Setup:** Download the `online_retail_II.csv` from Kaggle and place it in the root directory.

## 📋 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/DemandSense-AI.git](https://github.com/YOUR_USERNAME/DemandSense-AI.git)
   cd DemandSense-AI
