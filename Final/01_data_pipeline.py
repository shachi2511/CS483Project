import pandas as pd
import pandas_datareader.data as web
import datetime
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
GOLD_FILE = 'XAU_USD Historical Data.csv'
OUTPUT_FILE = os.path.join(DATA_DIR, 'master_dataset.csv')

# Expanded FRED Series IDs
FRED_SERIES = {
    'SAHMREALTIME': 'SAHMREALTIME', # Sahm Rule Recession Indicator
    'T10Y2Y': 'T10Y2Y',             # 10-Year Minus 2-Year Treasury Yield Spread
    'VIXCLS': 'VIXCLS',             # CBOE Volatility Index
    'DEXUSEU': 'DEXUSEU',           # US / Euro Exchange Rate
    'DTWEXBGS': 'DTWEXBGS',         # Nominal Broad US Dollar Index
    'CPIAUCSL': 'CPIAUCSL',         # Consumer Price Index (Inflation)
    'GDP': 'GDP',                   # Gross Domestic Product
    'PCEPI': 'PCEPI',               # Personal Consumption Expenditures
    'SP500': 'SP500',               # S&P 500 Index
    'DCOILWTICO': 'Crude_Oil'       # Crude Oil Prices (WTI)
}

def load_local_gold_data():
    """Loads and cleans the specific XAU_USD file."""
    path = os.path.join(DATA_DIR, GOLD_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    
    print(f"Loading local Gold data from {path}...")
    df = pd.read_csv(path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Price'].dtype == object:
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = df['Price'].astype(float)
    
    df = df.rename(columns={'Price': 'Gold_Price'})
    df = df.set_index('Date').sort_index()
    
    return df[['Gold_Price']]

def fetch_macro_data():
    """Fetches comprehensive economic data from FRED."""
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime.now()
    
    print("Fetching Expanded Macro data from FRED API...")
    try:
        # Fetch all series at once
        df = web.DataReader(list(FRED_SERIES.keys()), 'fred', start_date, end_date)
        
        # Rename columns to be friendlier
        df = df.rename(columns=FRED_SERIES)
        return df
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        return pd.DataFrame()

def main():
    # 1. Load Gold
    gold_df = load_local_gold_data()
    print(f"Gold Data Range: {gold_df.index.min().date()} to {gold_df.index.max().date()}")

    # 2. Load Macro
    macro_df = fetch_macro_data()
    
    # 3. Merge (Left Join to match Gold dates)
    print("Merging datasets...")
    merged_df = gold_df.join(macro_df, how='left')
    
    # 4. Fill Macro Gaps (Forward fill specifically for monthly/quarterly metrics like GDP)
    merged_df.ffill(inplace=True)
    
    # 5. Drop any remaining NaNs
    merged_df.dropna(inplace=True)
    
    # Save
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Success! Master dataset saved to {OUTPUT_FILE}")
    print(f"Columns: {list(merged_df.columns)}")
    print(merged_df.tail())

if __name__ == "__main__":
    main()