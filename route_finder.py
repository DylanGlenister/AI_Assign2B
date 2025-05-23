import construct_graph
import search


def find_route(_origin: int, _goal: int):
	# Need to pass information from model into this
	graph = construct_graph.create_graph('./processed.csv')
	problem = search.GraphProblem(_origin, _goal, graph)

	# Result is type search.Node
	result, _ = search.astar_search(problem, False)

	if result is None:
		return None

	path: list[search.Node] = result.solution()

	return path

def find_five_routes(_origin, _goal):
	return [
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal)
	]
