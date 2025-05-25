import folium as fm
import pandas as pd

import construct_graph
import shared
from RNNModels import PreLoadedPredictor


# Change to a list of 5 tuple with start, end, path, colour
def generate_map(_start: int, _end: int, _paths, _colours: list[tuple[str, str]]):

	locations = construct_graph.get_locations()

	start_pos = locations[_start]
	m = fm.Map(location=start_pos, zoom_start = 13)

	# Add markers to the start and end points
	fm.Marker(start_pos, popup="Start", icon=fm.Icon(color='blue')).add_to(m)
	fm.Marker(locations[_end], popup="End", icon=fm.Icon(color='red')).add_to(m)

	# At lines connecting all the nodes in the path
	for i, path in enumerate(_paths):
		prev = start_pos
		for node in path:
			position = locations[node]
			_, colour = _colours[i]
			fm.PolyLine(
				(prev, position),
				color=colour,
				weight=6
			).add_to(m)
			prev = position

	m.save(shared.PATH_ROUTEMAP)

def graph_visualisation(_model: str):
	'''Creates an openstreetmap file that displays the node/edge information.'''

	scats_df = pd.read_csv(shared.PATH_PROCESSED_DATASET)

	model = PreLoadedPredictor(_model, 0, '10:00')

	edge_lookup = construct_graph.get_edge_lookup()

	average_positions = construct_graph.get_locations()

	# Reduce the dataframe to only the needed unformation
	sites = scats_df[[shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE]].copy().drop_duplicates()

	site_locations: dict[tuple[int, str], tuple[float, float]] = {}

	m = fm.Map(location=(-37.82, 145.07), zoom_start=13)

	for _, row in sites.iterrows():
		scats: int = row[shared.COLUMN_SCAT]
		direction: str = row[shared.COLUMN_DIRECTION]
		latitude: float = row[shared.COLUMN_LATITUDE]
		longitude: float = row[shared.COLUMN_LONGITUDE]

		site_locations[(scats, direction)] = (latitude, longitude)

		# Find if a site is present in the lookup table
		present = any(end == scats and dir == direction for (end, dir), _ in edge_lookup.items())

		# Display the sites on the map, these are the edges
		fm.Marker(
			(latitude, longitude),
			tooltip=f'{scats}:{direction}',
			icon=fm.Icon(color='blue' if present else 'red')
		).add_to(m)

	def colour_lookup(cost):
		if cost > 200:
			return 'red'
		elif cost > 150:
			return 'orange'
		elif cost > 100:
			return 'yellow'
		elif cost > 75:
			return 'green'
		elif cost > 30:
			return 'blue'
		else:
			return 'purple'

	for (end, direction), start in edge_lookup.items():
		cost = model.query(end, direction)

		fm.PolyLine(
			locations=(site_locations[(end, direction)], average_positions[start]),
			color=colour_lookup(cost),
			weight=6
		).add_to(m)

	m.save(shared.PATH_GRAPHMAP)

def run_test():
	Start = -37.80486 + 0.00142, 145.08093 + 0.00171
	Connect = -37.82526 + 0.00142, 145.07758 + 0.00171
	End = -37.82371 + 0.00142, 145.06418 + 0.00171

	# Create the map
	m = fm.Map(location=Start, zoom_start = 15)
	fm.PolyLine((Start,Connect)).add_to(m)
	fm.PolyLine((Connect,End)).add_to(m)
	fm.Marker(End).add_to(m)

	# Display the map
	m.save('test_map.html')

if __name__ == '__main__':
	run_test()
