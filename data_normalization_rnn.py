import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


def normalise_data():
    min_max_scaler = MinMaxScaler()
    path = os.path.join('data', 'MAUS', 'Data', 'Raw_data')
    normalized_data_path = os.path.join('data', 'MAUS', 'Data', 'Normalized_data')
    os.makedirs(normalized_data_path, exisgit_ok=True)
    if not os.path.isdir(path):
        print(f"Raw data path does not exist: {path}")
        return
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            raw_file_path = os.path.join(folder_path, file)
            if not os.path.isfile(raw_file_path):
                continue
            dest_file = os.path.join(normalized_data_path, f'{folder_name}/{file}')
            if os.path.exists(dest_file):
                # already normalized
                continue
            try:
                if file.lower().endswith('.csv'):
                    df = pd.read_csv(raw_file_path)
                else:
                    df = pd.read_excel(raw_file_path)
            except Exception as e:
                print(f"Failed to read '{raw_file_path}': {e}")
                continue

            # keep only numeric columns for scaling
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty or numeric_df.shape[0] < 1:
                print(f"Skipping '{raw_file_path}': no numeric rows/columns to scale")
                continue

            try:
                scaled = min_max_scaler.fit_transform(numeric_df)
            except Exception as e:
                print(f"Scaling failed for '{raw_file_path}': {e}")
                continue

            scaled_df = pd.DataFrame(scaled, index=numeric_df.index, columns=numeric_df.columns)
            # If original had non-numeric columns, keep them as-is alongside scaled numeric cols
            non_numeric = df.drop(columns=numeric_df.columns, errors='ignore')
            if not non_numeric.empty:
                out_df = pd.concat([non_numeric.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
            else:
                out_df = scaled_df

            try:
                os.makedirs(os.path.join(normalized_data_path,folder_name), exist_ok=True)
                out_df.to_csv(dest_file, index=False)
            except Exception as e:
                print(f"Failed to write '{dest_file}': {e}")
            
        

if __name__ == "__main__":
    
    normalise_data()