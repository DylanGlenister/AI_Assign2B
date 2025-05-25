import matplotlib.pyplot as plt
import pandas as pd

import shared
from RNNModels import PreLoadedPredictor

df = pd.read_csv(shared.PATH_DATASET)
df.set_index([shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, 'ID', 'Day_of_week'], inplace=True)
df.drop(columns=[shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE], inplace=True)
df.drop_duplicates(inplace=True)

# Create a new dataframe that takes all the rows with the same SCAT, DIRECTION, and Day_of_week and average all the values together
averaged_df = df.groupby([shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, 'Day_of_week']).mean()

baseline_average = []
lstm_estimations = []
gru_estimations = []
tfmr_estimations = []
time_labels = []

for hour in range(0,24):
	for minute in range(0,46,15):
		time = f'{hour:0=2}:{minute:0=2}'
		time_labels.append(time)

		# Add the average for this location at this time
		baseline_average.append(averaged_df.loc[(3127, 'E', 0)][time])

		# Add all the models' estimations
		lstm_model = PreLoadedPredictor(model='LSTM', day_of_week=0, time_of_day=time)
		gru_model = PreLoadedPredictor(model='GRU', day_of_week=0, time_of_day=time)
		tfmr_model = PreLoadedPredictor(model='TFMR', day_of_week=0, time_of_day=time)
		lstm_estimations.append(lstm_model.query(3127, 'E'))
		gru_estimations.append(gru_model.query(3127, 'E'))
		tfmr_estimations.append(tfmr_model.query(3127, 'E'))

plt.figure(figsize=(12, 6))
# Plot all the estimations on a line graph over time
plt.plot(range(len(time_labels)), baseline_average, label='Baseline Average', linewidth=2, marker='o')
plt.plot(range(len(time_labels)), lstm_estimations, label='LSTM Predictions', linewidth=2, marker='s')
plt.plot(range(len(time_labels)), gru_estimations, label='GRU Predictions', linewidth=2, marker='^')
plt.plot(range(len(time_labels)), tfmr_estimations, label='Transformer Predictions', linewidth=2, marker='d')

plt.xlabel('Time of Day')
plt.ylabel('Traffic Volume')
plt.title('Traffic Prediction Models Comparison - SCAT: 3127, Direction: East, Day: Monday')
plt.legend()
plt.grid(True, alpha=0.3)

# Set x-axis labels to show every 4th time point for readability
plt.xticks(range(0, len(time_labels), 4), [time_labels[i] for i in range(0, len(time_labels), 4)], rotation=45)
plt.tight_layout()
plt.show()
