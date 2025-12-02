import pandas as pd
import pandas_datareader.data as web
import datetime
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
GOLD_FILE = 'XAU_USD Historical Data.csv'
OUTPUT_FILE = os.path.join(DATA_DIR, 'master_dataset.csv')

# Define FRED Series IDs
FRED_SERIES = {
    'SAHMREALTIME': 'SAHMREALTIME', # Sahm Rule Recession Indicator
    'T10Y2Y': 'T10Y2Y',             # 10-Year Minus 2-Year Treasury Yield Spread
    'VIXCLS': 'VIXCLS',             # CBOE Volatility Index
    'DEXUSEU': 'DEXUSEU',           # US / Euro Exchange Rate
    'DTWEXBGS': 'DTWEXBGS'          # Nominal Broad US Dollar Index
}

def load_local_gold_data():
    """Loads and cleans the specific XAU_USD file."""
    path = os.path.join(DATA_DIR, GOLD_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    
    print(f"Loading local Gold data from {path}...")
    df = pd.read_csv(path)
    
    # Cleaning: Handle comma in prices like "1,301.38"
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Price'].dtype == object:
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = df['Price'].astype(float)
    
    # Rename and Sort
    df = df.rename(columns={'Price': 'Gold_Price'})
    df = df.set_index('Date').sort_index()
    
    # Return ONLY the Gold Price column to avoid index duplication issues
    return df[['Gold_Price']]

def fetch_macro_data():
    """Fetches economic data from FRED."""
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime.now()
    
    print("Fetching Macro data from FRED API...")
    try:
        # Fetch all series at once
        df = web.DataReader(list(FRED_SERIES.keys()), 'fred', start_date, end_date)
        return df
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        # Return empty DF so pipeline doesn't crash, but warn user
        return pd.DataFrame()

def main():
    # 1. Load Gold
    gold_df = load_local_gold_data()
    print(f"Gold Data Range: {gold_df.index.min().date()} to {gold_df.index.max().date()}")
    print(f"Gold Rows: {len(gold_df)}")

    # 2. Load Macro
    macro_df = fetch_macro_data()
    
    # 3. Merge - FIX: Use 'left' join to preserve ONLY Gold Data dates
    print("Merging datasets (Left Join)...")
    merged_df = gold_df.join(macro_df, how='left')
    
    # 4. Fill Macro Gaps (Forward fill specifically for macro indicators)
    # We do this AFTER joining so we only fill gaps that exist within valid gold dates
    merged_df.ffill(inplace=True)
    
    # 5. Drop any remaining NaNs (e.g., start of dataset before macro data existed)
    merged_df.dropna(inplace=True)
    
    # 6. Sanity Check
    # Ensure the price isn't flat in the last 100 rows
    last_100_prices = merged_df['Gold_Price'].tail(100)
    if last_100_prices.nunique() <= 1:
        print("\nWARNING: The Gold Price seems to be constant (flat) at the end of the file.")
        print("This will cause the baseline model to have 0.00 error.")
        print("Please check your input CSV file.")
    else:
        print("\nData check passed: Prices are dynamic.")

    # Save
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Success! Master dataset saved to {OUTPUT_FILE}")
    print(f"Final Shape: {merged_df.shape}")
    print(merged_df.tail())

if __name__ == "__main__":
    main()