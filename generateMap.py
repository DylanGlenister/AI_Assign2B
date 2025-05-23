import folium as fm
import pandas as pd

import construct_graph
import shared


# Change to a list of 5 tuple with start, end, path, colour
def generate_map(_start: int, _end: int, _paths):

	locations = construct_graph.get_locations()

	start_pos = locations[_start]
	m = fm.Map(location=start_pos, zoom_start = 13)

	# Add markers to the start and end points
	fm.Marker(start_pos, popup="Start", icon=fm.Icon(color='blue')).add_to(m)
	fm.Marker(locations[_end], popup="End", icon=fm.Icon(color='red')).add_to(m)

	# At lines connecting all the nodes in the path
	for path in _paths:
		prev = start_pos
		for node in path:
			position = locations[node]
			fm.PolyLine((prev, position)).add_to(m)
			prev = position

	m.save(shared.PATH_ROUTEMAP)

def graph_visualisation():
	'''Creates an openstreetmap file that displays the node/edge information.'''

	scats_df = pd.read_csv(shared.PATH_DATASET)

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

	for end, start in edge_lookup.items():
		fm.PolyLine((site_locations[end], average_positions[start])).add_to(m)

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
