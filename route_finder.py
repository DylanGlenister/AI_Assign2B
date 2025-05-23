import construct_graph
import search


def find_route(_origin: int, _goal: int) -> None | list[search.Node]:
	# Need to pass information from model into this
	graph = construct_graph.create_graph()
	problem = search.GraphProblem(_origin, _goal, graph)

	# Result is type search.Node
	result, _ = search.astar_search(problem, False)

	return result.solution() if result is not None else None

def find_five_routes(_origin: int, _goal: int):
	# Ideas for 5 paths
	# 1. Default
	# 2. Ignore traffic, shortest distance
	# 3. Prioritise traffic, square the costs
	# 4. --- Please write an idea ---
	# 5. --- Please write an idea ---

	return [
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal),
		find_route(_origin, _goal)
	]
