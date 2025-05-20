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
def fix_burwood_auburn_latitude(_latitude: float):
	# Do it this funky way to avoid floating point nonsense
	if _latitude == 0:
		return -37.82505
	else:
		return _latitude

def fix_burwood_auburn_longitude(_longitude: float):
	if _longitude == 0:
		return 145.04346
	else:
		return _longitude

raw_data['Latitude'] = raw_data['Latitude'].apply(fix_burwood_auburn_latitude)
raw_data['Longitude'] = raw_data['Longitude'].apply(fix_burwood_auburn_longitude)

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
	#dates: list[str] = []
	#years: list[int] = []
	#months:list[int] = []
	#days: list[int] = []
	#day_indexes: list[int] = []
	days_of_week: list[str | None] = []

	for _, item in _dates.items():
		# Import as a datetime object
		date_obj = datetime.strptime(item, '%d/%m/%Y')
		#dates.append(date_obj.strftime("%Y-%m-%d"))
		#years.append(date_obj.year)
		#months.append(date_obj.month)
		#days.append(date_obj.day)
		#day_indexes.append((date_obj.date() - date(2000, 1, 1)).days)
		days_since_first_mon = (date_obj.date() - date(2000, 1, 3)).days % 7
		match days_since_first_mon:
			case 0:
				day_of_week = 'Monday'
			case 1:
				day_of_week = 'Tuesday'
			case 2:
				day_of_week = 'Wednesday'
			case 3:
				day_of_week = 'Thursday'
			case 4:
				day_of_week = 'Friday'
			case 5:
				day_of_week = 'Saturday'
			case 6:
				day_of_week = 'Sunday'
			case _:
				day_of_week = None
		days_of_week.append(day_of_week)

	#return dates, years, months, days, day_indexes, days_of_week
	return days_of_week

# The only one of value might be day of week, but as an int
#dates, years, months, days, date_indexes, days_of_week = process_date(extracted['Date'])
days_of_week = process_date(extracted['Date'])
#extracted['Date'] = dates
extracted.insert(8,'Day_of_week', days_of_week)
#extracted.insert(8, 'DayIndex', date_indexes)
#extracted.insert(8,'Day', days)
#extracted.insert(8,'Month', months)
#extracted.insert(8,'Year', years)

# Remove the location and date columns since they're no longer needed
extracted.drop(columns=['Location', 'Date'], inplace=True)

def reconfigure(_df: pd.DataFrame):
	# Create a list to store IDs
	ids = []

	# Process each location group
	for _, group in _df.groupby('Intersection', sort=False):
		# For each row in this location group, assign sequential IDs
		for i in range(len(group)):
			ids.append(i)

	# Add the IDs as a new column for use in the MultiIndex
	_df_with_ids = _df.copy()
	_df_with_ids['ID'] = ids

	# Create the MultiIndex
	index = pd.MultiIndex.from_arrays(
		[_df_with_ids['SCATS'], _df_with_ids['Intersection'], _df_with_ids['ID']],
		names=['SCATS', 'Intersection', 'ID']
	)

	# Drop the columns that are now in the index
	stripped = _df_with_ids.drop(columns=['SCATS', 'Intersection', 'ID'])
	stripped.set_index(index, inplace=True)

	return stripped

reconfigured = reconfigure(extracted)

# Save dataframe to csv
reconfigured.to_csv('./processed.csv')
