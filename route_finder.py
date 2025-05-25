import construct_graph
import search
from RNNModels import PreLoadedPredictor

# New functions should be created for the different routes

def find_route(_origin: int, _goal: int, _model: PreLoadedPredictor, _variant=0) -> None | list[search.Node]:
	'''Constructs a graph and pathfinds from the origin to the goal using a ML model to estimate traffic conditions.

		Parameters
		----------
		_origin : int
		_goal : int
		_time : str
		_model : str
		_variant : int, optional
			Changes how the graph is created.
			0. Default
			1. Ignore traffic, shortest distance
			2. Prioritise traffic, square the costs
			3. I hate traffic mode, cost to the fourth power
			4. Use depth first search
	'''

	print(f'Calculating route {_variant}')


	graph = construct_graph.create_graph(_model, _variant)
	problem = search.GraphProblem(_origin, _goal, graph)

	# Result is type search.Node
	if _variant == 4:
		result, _ = search.depth_first_graph_search(problem, False)
	else:
		result, _ = search.astar_search(problem, False)

	return result.solution() if result is not None else None

def find_five_routes(_origin: int, _goal: int, _time: str, _model: str):
	# Need to pass information from model into this
	model = PreLoadedPredictor(_model, 0, _time)
	return [
		# Reverse order to render bottom to top
		find_route(_origin, _goal, model, 4),
		find_route(_origin, _goal, model, 3),
		find_route(_origin, _goal, model, 2),
		find_route(_origin, _goal, model, 1),
		find_route(_origin, _goal, model, 0)
	]
