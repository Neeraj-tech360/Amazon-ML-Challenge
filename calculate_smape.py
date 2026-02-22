import pandas as pd
import numpy as np

def calculate_smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (pd.Series or np.array): Array of the first set of values.
        y_pred (pd.Series or np.array): Array of the second set of values.

    Returns:
        float: The SMAPE score as a percentage.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the numerator and denominator
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle the case where the denominator is zero to avoid division errors
    epsilon = 1e-8
    ratio = numerator / np.where(denominator == 0, epsilon, denominator)

    # Calculate the mean and convert to a percentage
    smape_score = np.mean(ratio) * 100
    return smape_score

def main():
    """
    Main function to load two prediction files, merge them, and calculate the SMAPE score between them.
    """
    # --- IMPORTANT: UPDATE THESE FILENAMES ---
    # Set the two prediction files you want to compare.
    file_a_path = 'yusufMurtaza.csv'
    file_b_path = '4_xgboost.csv'

    print("--- SMAPE Score Calculator (Comparing Two Prediction Files) ---")
    
    try:
        # Load the datasets
        print(f"Attempting to load File A from: '{file_a_path}'")
        df_a = pd.read_csv(file_a_path)

        print(f"Attempting to load File B from: '{file_b_path}'")
        df_b = pd.read_csv(file_b_path)

        print(f"\nLoaded {len(df_a)} rows from File A.")
        print(f"Loaded {len(df_b)} rows from File B.")

        # --- Column Validation ---
        for df, path in [(df_a, file_a_path), (df_b, file_b_path)]:
            if 'price' not in df.columns or 'sample_id' not in df.columns:
                print(f"\nError: The file '{path}' must contain 'sample_id' and 'price' columns.")
                return

        # Merge the two dataframes on 'sample_id' to align the prices
        merged_df = pd.merge(
            df_a[['sample_id', 'price']],
            df_b[['sample_id', 'price']],
            on='sample_id',
            suffixes=('_a', '_b')
        )

        if len(merged_df) == 0:
            print("\nError: No matching 'sample_id's found between the two files. Cannot calculate score.")
            print("Ensure both files contain the same set of sample IDs.")
            return

        print(f"\nSuccessfully merged {len(merged_df)} matching samples for comparison.")

        # Extract the price columns from both files
        prices_a = merged_df['price_a']
        prices_b = merged_df['price_b']

        # Calculate the SMAPE score
        smape_score = calculate_smape(prices_a, prices_b)

        # Print the final score, formatted to two decimal places
        print(f"\n=======================================================")
        print(f"  SMAPE Score between '{file_a_path}' and '{file_b_path}':")
        print(f"  {smape_score:.2f}%")
        print(f"=======================================================")
        print("(Lower values indicate the predictions are more similar)")


    except FileNotFoundError as e:
        print(f"\nError: Could not find a file. Please make sure both CSV files are in the same directory.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()