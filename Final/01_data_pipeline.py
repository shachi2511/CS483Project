# Combines FRED API + SAHM
# Outputs: master_dataset.csv
import pandas as pd
import pandas_datareader.data as web
import datetime
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
GOLD_FILE = 'XAU_USD Historical Data.csv'
OUTPUT_FILE = os.path.join(DATA_DIR, 'master_dataset.csv')

# Define FRED Series IDs to fetch (Combines logic from MACRO.ipynb + your SAHM request)
FRED_SERIES = {
    'SAHMREALTIME': 'SAHMREALTIME', # Sahm Rule Recession Indicator
    'T10Y2Y': 'T10Y2Y',             # 10-Year Minus 2-Year Treasury Yield Spread
    'VIXCLS': 'VIXCLS',             # CBOE Volatility Index (Market Risk)
    'DEXUSEU': 'DEXUSEU',           # US / Euro Exchange Rate
    'DTWEXBGS': 'DTWEXBGS'          # Nominal Broad US Dollar Index
}

def load_local_gold_data():
    """Loads and cleans the specific XAU_USD file from the ym1na folder."""
    path = os.path.join(DATA_DIR, GOLD_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Please move the CSV into the data folder.")
    
    print(f"Loading local Gold data from {path}...")
    df = pd.read_csv(path)
    
    # Cleaning based on standard investing.com format usually found in these projects
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    
    # Rename for clarity
    df = df.rename(columns={'Price': 'Gold_Price'})
    df = df.set_index('Date').sort_index()
    return df[['Gold_Price']]

def fetch_macro_data():
    """Fetches economic data including SAHM rule from FRED."""
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime.now()
    
    print("Fetching Macro data from FRED API...")
    try:
        # Fetch all series at once
        df = web.DataReader(list(FRED_SERIES.keys()), 'fred', start_date, end_date)
        return df
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        return pd.DataFrame()

def main():
    # 1. Load Gold
    gold_df = load_local_gold_data()
    
    # 2. Load Macro/Sahm Data
    macro_df = fetch_macro_data()
    
    # 3. Merge
    # We use 'outer' join to keep all dates, then sort
    print("Merging datasets...")
    merged_df = gold_df.join(macro_df, how='outer')
    
    # 4. Handling Missing Data (Crucial for Time Series)
    # Gold doesn't trade on weekends -> drop rows where Gold is NaN? 
    # OR ffill Macro data to match Gold trading days.
    # Strategy: Forward fill macro data (since economic stats don't change daily),
    # then drop days where we have no Gold price (weekends).
    merged_df.ffill(inplace=True)
    merged_df.dropna(subset=['Gold_Price'], inplace=True)
    
    # Save
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Success! Master dataset saved to {OUTPUT_FILE}")
    print(merged_df.head())

if __name__ == "__main__":
    main()