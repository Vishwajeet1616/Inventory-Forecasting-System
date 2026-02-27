from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor
import traceback 

app = Flask(__name__)

# --- 1. ROBUST DATA LOADING (CSV VERSION) ---
print("--- STARTING APP ---")
print("Loading CSV Dataset... Please wait...")

try:
    # TRY READING CSV with standard encoding first, if fail, try ISO-8859-1
    try:
        df = pd.read_csv('online_retail_II.csv')
    except UnicodeDecodeError:
        print("Standard UTF-8 failed. Trying ISO-8859-1 encoding...")
        df = pd.read_csv('online_retail_II.csv', encoding='ISO-8859-1')
    
    # FIX: Clean column names (Remove hidden spaces)
    df.columns = df.columns.str.strip()
    print("Detected Columns:", df.columns.tolist()) 

    # RENAME columns if they don't match exactly
    if 'StockCode' not in df.columns:
        for col in df.columns:
            if 'stock' in col.lower():
                df.rename(columns={col: 'StockCode'}, inplace=True)
                break
    
    # Keep only necessary columns
    df = df[['InvoiceDate', 'StockCode', 'Quantity']]
    
    # CLEANING 1: Remove returns
    df = df[df['Quantity'] > 0]
    
    # CLEANING 2: Convert Date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.set_index('InvoiceDate', inplace=True)
    
    # CLEANING 3: FORCE STOCK CODE TO STRING
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    
    print(f"Dataset Loaded! Found {len(df)} transactions.")
    
except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    traceback.print_exc()
    df = pd.DataFrame() 

# --- 2. PREDICTION FUNCTION ---
def get_prediction_and_plot(stock_code):
    try:
        # Search for the stock code
        stock_code = str(stock_code).strip()
        stock_data = df[df['StockCode'] == stock_code]
        
        print(f"Searching for: '{stock_code}' | Found {len(stock_data)} rows.")

        if stock_data.empty:
            return None, None, f"Stock Code '{stock_code}' not found."

        # Resample to Weekly
        weekly_data = stock_data['Quantity'].resample('W').sum().to_frame(name='Sales')
        weekly_data = weekly_data.fillna(0)
        
        # Feature Engineering
        for i in range(1, 5):
            weekly_data[f'Lag_{i}'] = weekly_data['Sales'].shift(i)
            
        weekly_data.dropna(inplace=True)
        
        if len(weekly_data) < 5:
            return None, None, "Not enough history (Need 5+ weeks)."

        # Train Model
        X = weekly_data[[f'Lag_{i}' for i in range(1, 5)]]
        y = weekly_data['Sales']
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Predict
        last_4_weeks = weekly_data['Sales'].iloc[-4:].values.reshape(1, -1)
        prediction = rf_model.predict(last_4_weeks)[0]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(weekly_data.index, weekly_data['Sales'], label='Historical Sales', color='black')
        
        next_date = weekly_data.index[-1] + pd.Timedelta(weeks=1)
        ax.scatter(next_date, prediction, color='red', s=150, label=f'Forecast: {int(prediction)}', zorder=5)
        
        ax.set_title(f"Forecast for {stock_code}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        return int(prediction), plot_url, None

    except Exception as e:
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"

# --- 3. WEB ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    plot_url = None
    curr_code = ""
    
    if request.method == 'POST':
        curr_code = request.form.get('stockcode', '')
        pred, plot, error = get_prediction_and_plot(curr_code)
        
        if error:
            prediction_text = error
        else:
            prediction_text = f"Predicted Demand: {pred} Units"
            plot_url = plot

    return render_template('index.html', prediction=prediction_text, plot_url=plot_url, curr_code=curr_code)

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
