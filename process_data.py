import re
from datetime import date, datetime

import numpy as np
import pandas as pd

import shared

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
	'SCATS Number': shared.COLUMN_SCAT,
	'NB_LATITUDE': shared.COLUMN_LATITUDE,
	'NB_LONGITUDE': shared.COLUMN_LONGITUDE,
}, inplace=True)

# SCATS is the intersection ID
# Location is [owner road] [direction from intersection] [other road in intersection]
# We only care about the direction

raw_data.drop_duplicates(inplace=True)

# Fix Auburn N/Burwood intersection missing position
def adjust_latitude(_latitude: float) -> float:
	# Fix Burwood/Aurburn latitude
	if _latitude == 0:
		return -37.82505 + 0.0015
	else:
		return _latitude + 0.0015

def adjust_longitude(_longitude: float) -> float:
	# Fix Burwood/Aurburn longitude
	if _longitude == 0:
		return 145.04346 + 0.0013
	else:
		return _longitude + 0.0013

raw_data[shared.COLUMN_LATITUDE] = raw_data[shared.COLUMN_LATITUDE].apply(adjust_latitude)
raw_data[shared.COLUMN_LONGITUDE] = raw_data[shared.COLUMN_LONGITUDE].apply(adjust_longitude)

# Import site reference
raw_reference = pd.read_csv(
	'./scats_reference.csv',
	names=[shared.COLUMN_SCAT, 'Intersection', 'Site_Type'],
	header=0,
	dtype={
		shared.COLUMN_SCAT: np.int32,
		'Intersection': str,
		'Site_Type': str
	}
)

raw_reference.drop_duplicates(inplace=True)
# Remove any site that isn't an intersection (rest are unused)
raw_reference = raw_reference[raw_reference.Site_Type == 'INT']
raw_reference.drop(columns={'Site_Type'}, inplace=True)

# Perform an inner merge to keep only SCATS sites present in both tables
merged_df = pd.merge(raw_reference, raw_data, on=shared.COLUMN_SCAT, how='inner')

# Extract location information
extracted = merged_df.copy()

def process_location(_locations: pd.Series) -> tuple[list[str], list[str]]:
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
extracted.insert(4, shared.COLUMN_DIRECTION, pd.Series(directions))

# Fix scat 4335 duplicated direction
mask = extracted[shared.COLUMN_LATITUDE] == -37.80474
extracted.loc[mask, shared.COLUMN_DIRECTION] = 'SE'

dated = extracted.copy()

def process_date(_dates: pd.Series) -> list[int]:
	days_of_week: list[int] = []

	for _, item in _dates.items():
		# Import as a datetime object
		date_obj = datetime.strptime(item, '%d/%m/%Y')
		days_since_first_mon = (date_obj.date() - date(2000, 1, 3)).days % 7

		days_of_week.append(days_since_first_mon)

	return days_of_week

days_of_week = process_date(dated['Date'])
dated.insert(8,'Day_of_week', days_of_week)

# Remove the location and date columns since they're no longer needed
dated.drop(columns=['Location', 'Date'], inplace=True)

def fix_times(_df: pd.DataFrame):
	changes = {}

	for hour in range(0,24):
		for minute in range(0,46,15):
			old = f'{hour}:{minute:0=2}'
			new = f'{hour:0=2}:{minute:0=2}'
			changes[old] = new

	return _df.rename(columns=changes)

retimed = fix_times(dated)

def reconfigure(_df: pd.DataFrame) -> pd.DataFrame:
	# Create sequential IDs within each group
	_df_with_ids = _df.copy()
	_df_with_ids['ID'] = _df_with_ids.groupby([shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, 'Day_of_week']).cumcount()

	# Create the MultiIndex
	_df_with_ids = _df_with_ids.set_index([shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, 'ID', 'Day_of_week'])

	return _df_with_ids

reconfigured = reconfigure(retimed)

def drop_pointless_sites(_df: pd.DataFrame) -> pd.DataFrame:
	'''Remove sites (edges) that do not connect to anything.'''
	to_drop = [
		(2827, 'N'),
		(2827, 'E'),
		(2827, 'W'),
		(4051, 'S'),
		(4030, 'W'),
		(3662, 'W'),
		(4821, 'N'),
		(4821, 'W'),
		(4821, 'S'),
		(4812, 'S'),
		(4812, 'SW'),
		(4812, 'NE'),
		(4270, 'W'),
		(4270, 'S'),
		(3180, 'E'),
		(4057, 'E'),
		(2200, 'N'),
		(2200, 'E'),
		(2200, 'S'),
		(3126, 'E'),
		(3682, 'E'),
		(2000, 'E'),
		(3685, 'E'),
		(970, 'E'),
		(970, 'S'),
		(2846, 'N'),
		(2846, 'NW'),
		(2846, 'W'),
		(4043, 'S'),
		(3812, 'W'),
		(3812, 'SE'),
		(4043, 'S'),
		(4035, 'E'),
		(3120, 'W'),
		(4263, 'S'),
		(4266, 'N'),
		(4266, 'S')
	]

	# Sort the index if it isn't already sorted
	if not _df.index.is_monotonic_increasing:
		_df = _df.sort_index()

	return _df.drop(index=to_drop)

reduced = drop_pointless_sites(reconfigured)

# Save dataframe to csv
reduced.to_csv(shared.PATH_DATASET)
