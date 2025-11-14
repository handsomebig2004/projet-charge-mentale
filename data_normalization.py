import pandas as pd
from sklearn.preprocessing import normalize
import os


def normalise_data():
    path = 'data/MAUS/Data/Raw_data'
    normalized_data_path = 'data/MAUS/Data/Normalied_data'
    for folder_name in os.listdir(path):
        for file in os.listdir(path + '/' + folder_name):
            if file[-3:] == 'csv':
                df = pd.read_csv(f'{path}/{folder_name}/{file}')
            else:
                df = pd.read_excel(f'{path}/{folder_name}/{file}')
            df = normalize(df) #TODO this transforms into a numpy array (so no to_csv)
            df.to_csv(f'{normalized_data_path}/{folder_name}/normalized_{file}')
            
        

if __name__ == "__main__":
    
    normalise_data()