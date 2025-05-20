from collections import defaultdict

import pandas as pd

import search


def create_graph(_df: pd.DataFrame):
	'''Take the information from a dataframe and create a graph from it.'''

	locations: dict[int, tuple[float, float]] = {}
	street_to_nodes: dict[str, list[int]] = {}

	for _, row in _df.iterrows():
		scats_num: int = row['SCATS']
		latitude: float = row['Latitude']
		longitude: float = row['Longitude']
		loc_desc: str = row['Intersection']

		# Locations is easy to set up
		locations[scats_num] = (latitude, longitude)

		# Split the location description by '/' to get individual streets
		# Clean and process each street name
		streets = [street.strip() for street in loc_desc.split('/')]

		# Associate each street with the SCATS number
		for street in streets:
			if street:
				if street not in street_to_nodes:
					street_to_nodes[street] = []
				street_to_nodes[street].append(scats_num)

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

	return locations, edges

processed_data = pd.read_csv('./processed.csv')

locations, edges = create_graph(processed_data)

# Print basic graph info
print(f'Number of nodes (intersections): {len(locations)}\nNumber of edges (street connections): {len(edges)}')

# List all edges and their cost
for node, others in edges.items():
	lat, long = locations[node]
	print(f'{node:4} ({lat:.6f}, {long:.6f}): {{', end='')
	for other, cost in others.items():
		print(f'{other:4}: {cost}, ', end='')
	print('}')


method = search.select_method('DFS')

if method is None:
	print("Incorrect method type, valid methods:\nDFS, BFS, GBFS, AS, CUS1, CUS2, IDS, BS")
	quit()

graph = search.Graph(edges)
graph.locations = locations

origin = 4030
goal = [970]

problem = search.GraphProblem(origin, goal, graph)

result, count = method(problem, True)

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
