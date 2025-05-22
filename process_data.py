import re
from datetime import date, datetime

import numpy as np
import pandas as pd

# Import dataset
raw_data = pd.read_csv(
	'./scats_data.csv',
	dtype={
		'SCATS Number': int,
		'Location': str,
		'NB_LATITUDE': float,
		'NB_LONGITUDE': float
	}
)

# Rename select columns
raw_data.rename(columns={
	'SCATS Number': 'SCATS',
	'NB_LATITUDE': 'Latitude',
	'NB_LONGITUDE': 'Longitude',
}, inplace=True)

# SCATS is the intersection ID
# Location is [owner road] [direction from intersection] [other road in intersection]

raw_data.drop_duplicates(inplace=True)

# Fix Auburn N/Burwood intersection missing position
# https://www.openstreetmap.org/way/1092802786#map=19/-37.823687/145.045020
# south: -37.82542, 145.04346
# east: -37.82529, 145.04387
# west: -37.82518, 145.04301
# north: -37.82505, 145.04346 (estimated by Claude)
def adjust_latitude(_latitude: float):
	# Fix Burwood/Aurburn latitude
	if _latitude == 0:
		return -37.82505 + 0.00142
	else:
		return _latitude + 0.00142

def adjust_longitude(_longitude: float):
	# Fix Burwood/Aurburn longitude
	if _longitude == 0:
		return 145.04346 + 0.00171
	else:
		return _longitude + 0.00171

raw_data['Latitude'] = raw_data['Latitude'].apply(adjust_latitude)
raw_data['Longitude'] = raw_data['Longitude'].apply(adjust_longitude)

# Import site reference
raw_reference = pd.read_csv(
	'./scats_reference.csv',
	names=['SCATS', 'Intersection', 'Site_Type'],
	header=0,
	dtype={
		'SCATS': np.int32,
		'Intersection': str,
		'Site_Type': str
	}
)

raw_reference.drop_duplicates(inplace=True)
# Remove any site that isn't an intersection (rest are unused)
raw_reference = raw_reference[raw_reference.Site_Type == 'INT']
raw_reference.drop(columns={'Site_Type'}, inplace=True)

# Perform an inner merge to keep only SCATS sites present in both tables
merged_df = pd.merge(raw_reference, raw_data, on='SCATS', how='inner')

# Extract location information
extracted = merged_df.copy()

def process_location(_locations: pd.Series):
	streets: list[str] = []
	directions: list[str] = []

	for _, item in _locations.items():
		parts: list[str] = re.split(' of ', item, flags=re.IGNORECASE)
		first_part = parts[0]

		# Get all words in the first part
		words = first_part.split()

		# Last word is the direction, everything before is the street
		direction = words[-1]
		street = ' '.join(words[:-1])

		streets.append(street)
		directions.append(direction)

	return streets, directions

streets, directions = process_location(extracted['Location'])
extracted.insert(3, 'Street', pd.Series(streets))
extracted.insert(4, 'Direction', pd.Series(directions))

def process_date(_dates: pd.Series):
	days_of_week = []

	for _, item in _dates.items():
		# Import as a datetime object
		date_obj = datetime.strptime(item, '%d/%m/%Y')
		days_since_first_mon = (date_obj.date() - date(2000, 1, 3)).days % 7

		days_of_week.append(days_since_first_mon)

	#return dates, years, months, days, day_indexes, days_of_week
	return days_of_week

days_of_week = process_date(extracted['Date'])
extracted.insert(8,'Day_of_week', days_of_week)

# Remove the location and date columns since they're no longer needed
extracted.drop(columns=['Location', 'Date'], inplace=True)

def reconfigure(_df: pd.DataFrame):
	# Create sequential IDs within each group
	_df_with_ids = _df.copy()
	_df_with_ids['ID'] = _df_with_ids.groupby(['SCATS', 'Direction', 'Day_of_week']).cumcount()

	# Create the MultiIndex
	_df_with_ids = _df_with_ids.set_index(['SCATS', 'Direction', 'Day_of_week', 'ID'])

	return _df_with_ids

reconfigured = reconfigure(extracted)

# Save dataframe to csv
reconfigured.to_csv('./processed.csv')
