from collections import defaultdict

import pandas as pd
import folium as fm

import search

def create_graph(_df: pd.DataFrame, _debug = False):
	'''Programmatically uses the information from the dataset to construct the graph.'''

	edge_lookup: dict[tuple[int, str], int] = {}
	with open('./edge_lookup.txt', 'r') as file:
		for line in file.readlines():
			end, dir, start = line.strip().split(',')
			edge_lookup[(int(end), dir)] = int(start)

	m = fm.Map(location=(-37.82, 145.07), zoom_start=13)

	# Reduce the dataframe to only the needed unformation
	sites = _df[['SCATS', 'Intersection', 'Direction', 'Latitude', 'Longitude']].copy().drop_duplicates()

	#scat_locations: dict[int, tuple[float, float]] = {}
	site_locations: dict[tuple[int, str], tuple[float, float]] = {}
	street_to_nodes: dict[str, list[int]] = {}

	for _, row in sites.iterrows():
		scats: int = row['SCATS']
		latitude: float = row['Latitude']
		longitude: float = row['Longitude']
		intersection: str = row['Intersection']
		direction: str = row['Direction']

		# Locations is easy to set up
		#scat_locations[scats] = (latitude, longitude)

		site_locations[(scats, direction)] = (latitude, longitude)

		# Find if a site is present in the lookup table
		present = any(end == scats and dir == direction for (end, dir), start in edge_lookup.items())

		# Display the sites on the map, these are the edges
		fm.Marker(
			(latitude, longitude),
			tooltip=f'{scats}:{direction}',
			icon=fm.Icon(color='red' if present else 'blue')
		).add_to(m)

		streets = [street.strip() for street in intersection.split('/')]

		# Associate each street with the SCATS number
		for street in streets:
			if street:
				if street not in street_to_nodes:
					street_to_nodes[street] = []
				street_to_nodes[street].append(scats)

	# Add the edges to the graph

	# Average the locations
	grouped = sites[['SCATS', 'Latitude', 'Longitude']].groupby('SCATS').mean()
	average_position = {scat: (pos['Latitude'], pos['Longitude']) for scat, pos in grouped.iterrows()}

	for end, start in edge_lookup.items():
		fm.PolyLine((site_locations[end], average_position[start])).add_to(m)

	m.save('graph.html')

	edge_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for {node: {connected_node: cost}}

	# Connect a node to all other nodes with the same street
	for _, nodes in street_to_nodes.items():
		for node in nodes:
			# Add all other nodes from this street as edges with default cost 1
			for connected_node in nodes:
				if connected_node != node:
					edge_dict[node][connected_node] = 1

	# Convert to a regular dictionary
	edges: dict[int, dict[int, int]] = {node: dict(connected_nodes) for node, connected_nodes in edge_dict.items()}

	if _debug:
		# Print basic graph info
		print(f'Number of nodes (intersections): {len(average_position)}\nNumber of edges (street connections): {len(edges)}')

		# List all edges and their cost
		for node, others in edges.items():
			lat, long = average_position[node]
			print(f'{node:4} ({lat:.6f}, {long:.6f}): {{', end='')
			for other, cost in others.items():
				print(f'{other:4}: {cost}, ', end='')
			print('}')

	graph = search.Graph(edges)
	graph.locations = average_position

	return graph, (edges, average_position)

def test():
	processed_data = pd.read_csv('./processed.csv')

	graph, _ = create_graph(processed_data)

	origin = 4030
	goal = [970]

	problem = search.GraphProblem(origin, goal, graph)

	result, count = search.astar_search(problem, False)

	print('method=AS')
	# \n
	# Ouput goal node
	print('goal=', goal, sep='', end=' | ')

	# Output number (length of path)
	print('number of nodes=', count, sep='')
	# \n
	if (result is not None):
		# Output path: list of nodes
		print('path=', result.solution(), sep='')
	else:
		print('No path found!')

if __name__ == '__main__':
	test()
