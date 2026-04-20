import pandas as pd

def get_data():
  path = kagglehub.dataset_download("WiDS-Global-Datathon")
  data = pd.read_csv(path + '/train.csv')
  return data

