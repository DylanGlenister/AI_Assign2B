import construct_graph
import search
from RNNModels import PreLoadedPredictor

# New functions should be created for the different routes

def find_route(_origin: int, _goal: int, _time: str, _model: str) -> None | list[search.Node]:
	# Need to pass information from model into this
	model = PreLoadedPredictor(_model, 0, _time)

	graph = construct_graph.create_graph(model)
	problem = search.GraphProblem(_origin, _goal, graph)

	# Result is type search.Node
	result, _ = search.astar_search(problem, False)

	return result.solution() if result is not None else None

def find_five_routes(_origin: int, _goal: int, _time: str, _model: str):
	# Ideas for 5 paths
	# 1. Default
	# 2. Ignore traffic, shortest distance
	# 3. Prioritise traffic, square the costs
	# 4. --- Please write an idea ---
	# 5. --- Please write an idea ---

	return [
		find_route(_origin, _goal, _time, _model),
		find_route(_origin, _goal, _time, _model),
		find_route(_origin, _goal, _time, _model),
		find_route(_origin, _goal, _time, _model),
		find_route(_origin, _goal, _time, _model)
	]
