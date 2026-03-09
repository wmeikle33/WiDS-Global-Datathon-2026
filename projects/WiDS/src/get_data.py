import pandas as pd

path = kagglehub.dataset_download("mathchi/diabetes-data-set")
data = pd.read_csv(path + '/diabetes.csv')

