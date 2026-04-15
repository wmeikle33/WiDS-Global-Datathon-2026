import pandas as pd

def get_data():
  path = kagglehub.dataset_download("WiDS-Global-Datathon/diabetes-data-set")
  data = pd.read_csv(path + '/diabetes.csv')
  return data

