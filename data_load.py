import pandas as pd

def load_csv(file):
    path = ('data/'+file+'.csv')
    data_file = pd.read_csv(path)
    return (data_file)