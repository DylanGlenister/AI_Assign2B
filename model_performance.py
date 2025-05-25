import matplotlib.pyplot as plt
import pandas as pd

import shared
from RNNModels import PreLoadedPredictor

scats_df = pd.read_csv(shared.PATH_DATASET)
# Reduce the dataframe to only the needed unformation
sites = scats_df[[shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE]].copy().drop_duplicates()

for hour in range(0,24):
	for minute in range(0,46,15):
		time = f'{hour:0=2}:{minute:0=2}'
		lstm_model = PreLoadedPredictor('LSTM', 0, time)
		gru_model = PreLoadedPredictor('GRU', 0, time)
		tfmr_model = PreLoadedPredictor('TFMR', 0, time)
		lstm_model.query(3127, 'E')
		gru_model.query(3127, 'E')
		tfmr_model.query(3127, 'E')


plt.figure(figsize=(12, 6))
