import folium as fm
import pandas as pd

import search


def get_edge_lookup():
	'''Retrieves the data from the edge lookup file. Format is {(end, dir), start}.'''
	edge_lookup: dict[tuple[int, str], int] = {}
	with open('./edge_lookup.txt', 'r') as file:
		for line in file.readlines():
			end, dir, start = line.strip().split(',')
			edge_lookup[(int(end), dir)] = int(start)
	return edge_lookup

def graph_visualisation(_df: pd.DataFrame):
	'''Creates an openstreetmap file that displays the node/edge information.'''

	edge_lookup = get_edge_lookup()

	m = fm.Map(location=(-37.82, 145.07), zoom_start=13)

	# Reduce the dataframe to only the needed unformation
	sites = _df[['SCATS', 'Direction', 'Latitude', 'Longitude']].copy().drop_duplicates()

	site_locations: dict[tuple[int, str], tuple[float, float]] = {}

	for _, row in sites.iterrows():
		scats: int = row['SCATS']
		latitude: float = row['Latitude']
		longitude: float = row['Longitude']
		direction: str = row['Direction']

		site_locations[(scats, direction)] = (latitude, longitude)

		# Find if a site is present in the lookup table
		present = any(end == scats and dir == direction for (end, dir), _ in edge_lookup.items())

		# Display the sites on the map, these are the edges
		fm.Marker(
			(latitude, longitude),
			tooltip=f'{scats}:{direction}',
			icon=fm.Icon(color='red' if present else 'blue')
		).add_to(m)

	grouped = sites[['SCATS', 'Latitude', 'Longitude']].groupby('SCATS').mean()
	average_position = {scat: (pos['Latitude'], pos['Longitude']) for scat, pos in grouped.iterrows()}

	for end, start in edge_lookup.items():
		fm.PolyLine((site_locations[end], average_position[start])).add_to(m)

	m.save('graph.html')

def create_graph(_df: pd.DataFrame, _debug = False):
	'''Programmatically uses the information from the dataset to construct the graph.'''

	edge_lookup = get_edge_lookup()

	# Reduce the dataframe to only the needed unformation
	sites = _df[['SCATS', 'Direction', 'Latitude', 'Longitude']].copy().drop_duplicates()

	# Average the locations
	grouped = sites[['SCATS', 'Latitude', 'Longitude']].groupby('SCATS').mean()
	average_position = {scat: (pos['Latitude'], pos['Longitude']) for scat, pos in grouped.iterrows()}

	# Edge lookup will be used to query the model for the cost

	# Add the edges to the graph
	# Edges need to be in the format {start: {end, cost}}

	edges: dict[int, dict[int, int]] = {}

	for (end, _), start in edge_lookup.items():
		if start not in edges:
			edges[start] = {}
		edges[start][end] = 1

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
	graph_visualisation(processed_data)

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
