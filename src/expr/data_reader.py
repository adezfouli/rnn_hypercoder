import h5py
import joblib as joblib
import pandas as pd
import numpy as np

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_BD():
        data = pd.read_csv("../data/BD/choices_diagno.csv.zip", compression='zip', header=0, sep=',', quotechar='"', keep_default_na=False)
        data['reward'] = [0 if x == 'null' else 1 for x in data['outcome']]
        data['id'] = data['ID']
        data['block'] = data['trial']

        #R1: right, R2: Left
        data['action'] = [0 if x == 'R1' else 1 for x in data['key']]
        return data


    @staticmethod
    def read_synth_normal():
        data = pd.read_csv("../data/synth/normal.csv", header=0, sep=',', quotechar='"', keep_default_na=False)
        data['block'] = 1
        return data