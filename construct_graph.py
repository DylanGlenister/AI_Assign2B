from typing import cast

import pandas as pd

import search
import shared
from RNNModels import PreLoadedPredictor


def get_edge_lookup() -> dict[tuple[int, str], int]:
	'''Retrieves the data from the edge lookup file. Format is {(end, dir), start}.'''
	edge_lookup: dict[tuple[int, str], int] = {}
	with open(shared.PATH_EDGE_LOOKUP, 'r') as file:
		for line in file.readlines():
			end, dir, start = line.strip().split(',')
			edge_lookup[(int(end), dir)] = int(start)
	return edge_lookup

def get_locations() -> dict[int, tuple[float, float]]:
	'''Get a dictionary of all SCATs sites and their locations.'''
	scats_df = pd.read_csv(shared.PATH_DATASET)
	# Reduce the dataframe to only the needed unformation
	sites = scats_df[[shared.COLUMN_SCAT, shared.COLUMN_DIRECTION, shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE]].copy().drop_duplicates()
	# Average the locations
	grouped = sites[[shared.COLUMN_SCAT, shared.COLUMN_LATITUDE, shared.COLUMN_LONGITUDE]].groupby(shared.COLUMN_SCAT).mean()
	# Need to do it this way to force the type checker into calming down
	average_positions = {
		int(scat_id): (
			cast(float, grouped.loc[scat_id, shared.COLUMN_LATITUDE]),
			cast(float, grouped.loc[scat_id, shared.COLUMN_LONGITUDE])
		)
		for scat_id in grouped.index
	}
	return average_positions

def create_graph(_model: PreLoadedPredictor, _debug = False) -> search.Graph:
	'''Programmatically uses the information from the dataset to construct the graph.'''

	edge_lookup = get_edge_lookup()

	# Edge lookup will be used to query the model for the cost

	# Add the edges to the graph
	# Edges need to be in the format {start: {end, cost}}
	edges: dict[int, dict[int, int]] = {}

	for (end, direction), start in edge_lookup.items():
		if start not in edges:
			edges[start] = {}
		edges[start][end] = _model.query(end, direction)

	average_positions = get_locations()

	if _debug:
		# Print basic graph info
		print(f'Number of nodes (intersections): {len(average_positions)}\nNumber of edges (street connections): {len(edges)}')

		# List all edges and their cost
		for node, others in edges.items():
			lat, long = average_positions[node]
			print(f'{node:4} ({lat:.6f}, {long:.6f}): {{', end='')
			for other, cost in others.items():
				print(f'{other:4}: {cost}, ', end='')
			print('}')

	graph = search.Graph(edges)
	graph.locations = average_positions

	return graph

def test():
	dummy_model = PreLoadedPredictor('LSTM', 0, '8:30')
	graph = create_graph(dummy_model)

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
