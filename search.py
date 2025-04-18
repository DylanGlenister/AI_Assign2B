"""
Assignment 2A Pathfinding algorithms.

Uses code from AIMA Chapters 3-4
"""

from collections import deque
import sys
import functools
import heapq
import re
import time
import numpy as np

def is_in(elt, seq):
	"""Similar to (elt in seq), but compares with 'is', not '=='. From AIMA."""

	return any(x is elt for x in seq)

def distance(a: tuple, b: tuple):
	"""The distance between two (x, y) points. From AIMA."""

	xA, yA = a
	xB, yB = b
	return np.hypot((xA - xB), (yA - yB))

def memoize(fn, slot=None, maxsize=32):
	"""Memoize fn: make it remember the computed value for any argument list.
	If slot is specified, store result in that slot of first argument.
	If slot is false, use lru_cache for caching the values. From AIMA."""

	# BUG Trying to use the slot causes a TypeError. Unable to fix it.
	if slot:
		def memoized_normal(obj, *args):
			if hasattr(obj, slot):
				return getattr(obj, slot)
			else:
				val = fn(obj, *args)
				setattr(obj, slot, val)
				return val
		out = memoized_normal
	else:
		@functools.lru_cache(maxsize=maxsize)
		def memoized_cached(*args):
			return fn(*args)
		out = memoized_cached

	return out

class PriorityQueue:
	"""A Queue in which the minimum (or maximum) element (as determined by f and
	order) is returned first.
	If order is 'min', the item with minimum f(x) is
	returned first; if order is 'max', then it is the item with maximum f(x).
	Also supports dict-like lookup. From AIMA."""

	def __init__(self, order='min', f=lambda x: x):
		self.heap = []
		if order == 'min':
			self.f = f
		elif order == 'max':  # now item with max f(x)
			self.f = lambda x: -f(x)  # will be popped first
		else:
			raise ValueError("Order must be either 'min' or 'max'.")

	def append(self, item):
		"""Insert item at its correct position."""
		heapq.heappush(self.heap, (self.f(item), item))

	def extend(self, items):
		"""Insert each item in items at its correct position."""
		for item in items:
			self.append(item)

	def pop(self):
		"""Pop and return the item (with min or max f(x) value)
		depending on the order."""
		if self.heap:
			return heapq.heappop(self.heap)[1]
		else:
			raise Exception('Trying to pop from empty PriorityQueue.')

	def __len__(self):
		"""Return current capacity of PriorityQueue."""
		return len(self.heap)

	def __contains__(self, key):
		"""Return True if the key is in PriorityQueue."""
		return any([item == key for _, item in self.heap])

	def __getitem__(self, key):
		"""Returns the first value associated with key in PriorityQueue.
		Raises KeyError if key is not present."""
		for value, item in self.heap:
			if item == key:
				return value
		raise KeyError(str(key) + " is not in the priority queue")

	def __delitem__(self, key):
		"""Delete the first occurrence of key."""
		try:
			del self.heap[[item == key for _, item in self.heap].index(True)]
		except ValueError:
			raise KeyError(str(key) + " is not in the priority queue")
		heapq.heapify(self.heap)

class Problem:
	"""The abstract class for a formal problem. You should subclass
	this and implement the methods actions and result, and possibly
	__init__, goal_test, and path_cost. Then you will create instances
	of your subclass and solve them with the various search functions.
	From AIMA."""

	def __init__(self, initial, goal=None):
		"""The constructor specifies the initial state, and possibly a goal
		state, if there is a unique goal. Your subclass's constructor can add
		other arguments."""
		self.initial = initial
		self.goal = goal

	def actions(self, state):
		"""Return the actions that can be executed in the given
		state. The result would typically be a list, but if there are
		many actions, consider yielding them one at a time in an
		iterator, rather than building them all at once."""
		raise NotImplementedError

	def result(self, state, action):
		"""Return the state that results from executing the given
		action in the given state. The action must be one of
		self.actions(state)."""
		raise NotImplementedError

	def goal_test(self, state):
		"""Return True if the state is a goal. The default method compares the
		state to self.goal or checks for state in self.goal if it is a
		list, as specified in the constructor. Override this method if
		checking against a single self.goal is not enough."""
		if isinstance(self.goal, list):
			return is_in(state, self.goal)
		else:
			return state == self.goal

	def path_cost(self, cost, state1, action, state2):
		"""Return the cost of a solution path that arrives at state2 from
		state1 via action, assuming cost c to get up to state1. If the problem
		is such that the path doesn't matter, this function will only look at
		state2. If the path does matter, it will consider c and maybe state1
		and action. The default method costs 1 for every step in the path."""
		return cost + 1

	def value(self, state):
		"""For optimization problems, each state has a value. Hill Climbing
		and related algorithms try to maximize this value."""
		raise NotImplementedError

class Node:
	"""A node in a search tree. Contains a pointer to the parent (the node
	that this is a successor of) and to the actual state for this node. Note
	that if a state is arrived at by two paths, then there are two nodes with
	the same state. Also includes the action that got us to this state, and
	the total path_cost (also known as g) to reach the node. Other functions
	may add an f and h value; see best_first_graph_search and astar_search for
	an explanation of how the f and h values are handled. You will not need to
	subclass this class. From AIMA."""

	f = None

	def __init__(self, state, parent=None, action=None, path_cost=0):
		"""Create a search tree Node, derived from a parent by an action."""
		self.state = state
		self.parent = parent
		self.action = action
		self.path_cost = path_cost
		self.depth = 0
		if parent:
			self.depth = parent.depth + 1

	def __repr__(self):
		return "<Node {}>".format(self.state)

	def __lt__(self, node):
		return self.state < node.state

	def expand(self, problem):
		"""List the nodes reachable in one step from this node."""
		return [self.child_node(problem, action)
				for action in problem.actions(self.state)]

	def child_node(self, problem, action):
		"""[Figure 3.10]"""
		next_state = problem.result(self.state, action)
		next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
		return next_node

	def solution(self):
		"""Return the sequence of actions to go from the root to this node."""
		return [node.action for node in self.path()[1:]]

	def path(self):
		"""Return a list of nodes forming the path from the root to this node."""
		node, path_back = self, []
		while node:
			path_back.append(node)
			node = node.parent
		return list(reversed(path_back))

	# We want for a queue of nodes in breadth_first_graph_search or
	# astar_search to have no duplicated states, so we treat nodes
	# with the same state as equal. [Problem: this may not be what you
	# want in other contexts.]

	def __eq__(self, other):
		return isinstance(other, Node) and self.state == other.state

	def __hash__(self):
		# We use the hash value of the state
		# stored in the node instead of the node
		# object itself to quickly search a node
		# with the same state in a Hash Table
		return hash(self.state)

class Graph:
	"""A graph connects nodes (vertices) by edges (links). Each edge can also
	have a length associated with it. The constructor call is something like:
		g = Graph({'A': {'B': 1, 'C': 2})
	this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
	A to B,  and an edge of length 2 from A to C. You can also do:
		g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
	This makes an undirected graph, so inverse links are also added. The graph
	stays undirected; if you add more links with g.connect('B', 'C', 3), then
	inverse link is also added. You can use g.nodes() to get a list of nodes,
	g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
	length of the link from A to B. 'Lengths' can actually be any object at
	all, and nodes can be any hashable object. From AIMA."""

	def __init__(self, graph_dict=None, directed=True):
		self.locations = {}
		self.least_costs = {}
		self.graph_dict = graph_dict or {}
		self.directed = directed
		if not directed:
			self.make_undirected()

	def make_undirected(self):
		"""Make a digraph into an undirected graph by adding symmetric edges."""
		for a in list(self.graph_dict.keys()):
			for (b, dist) in self.graph_dict[a].items():
				self.connect1(b, a, dist)

	def connect(self, A, B, distance=1):
		"""Add a link from A and B of given distance, and also add the inverse
		link if the graph is undirected."""
		self.connect1(A, B, distance)
		if not self.directed:
			self.connect1(B, A, distance)

	def connect1(self, A, B, distance):
		"""Add a link from A to B of given distance, in one direction only."""
		self.graph_dict.setdefault(A, {})[B] = distance

	def get(self, a, b=None):
		"""Return a link distance or a dict of {node: distance} entries.
		.get(a,b) returns the distance or None;
		.get(a) returns a dict of {node: distance} entries, possibly {}."""
		links = self.graph_dict.setdefault(a, {})
		if b is None:
			return links
		else:
			return links.get(b)

	def nodes(self):
		"""Return a list of nodes in the graph."""
		s1 = set([k for k in self.graph_dict.keys()])
		s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
		nodes = s1.union(s2)
		return list(nodes)

class GraphProblem(Problem):
	"""The problem of searching a graph from one node to another. From AIMA."""

	def __init__(self, initial, goal, graph):
		super().__init__(initial, goal)
		self.graph = graph

	def actions(self, state):
		"""The actions at a graph node are just its neighbors."""
		return list(self.graph.get(state).keys())

	def result(self, state, action):
		"""The result of going to a neighbor is just that neighbor."""
		return action

	def path_cost(self, cost, state1, action, state2):
		"""Cost is the cost so far."""
		return cost + (self.graph.get(state1, state2) or np.inf)

	def find_min_edge(self):
		"""Find minimum value of edges."""
		m = np.inf
		for d in self.graph.graph_dict.values():
			local_min = min(d.values())
			m = min(m, local_min)

		return m

	def h(self, node):
		"""h function is straight-line distance from a node's state to goal."""
		#locs = getattr(self.graph, 'locations', None)
		locs = self.graph.locations
		if locs:
			# Fix multiple destination nodes
			if isinstance(self.goal, list):
				shortest = (None, np.inf)
				for goal in self.goal:
					node_dist = int(distance(locs[node.state], locs[goal]))
					if node_dist < shortest[1]:
						shortest = (node, node_dist)
				return shortest[1]
			else:
				if type(node) is str:
					return int(distance(locs[node], locs[self.goal]))

				return int(distance(locs[node.state], locs[self.goal]))
		else:
			return np.inf

def depth_first_graph_search(_problem: GraphProblem, _debug) -> tuple[Node | None, int]:
	"""Search the deepest nodes in the search tree first.
	Search through the successors of a problem to find a goal.
	The argument frontier should be an empty queue.
	Does not get trapped by loops.
	If two paths reach a state, only use the first one.
	From AIMA.
	"""
	frontier = [(Node(_problem.initial))]  # Stack

	explored = set()
	while frontier:
		node = frontier.pop()
		if _debug:
			print(node, node.expand(_problem))
		if _problem.goal_test(node.state):
			return node, len(explored)
		explored.add(node.state)
		frontier.extend(child for child in node.expand(_problem)
						if child.state not in explored and child not in frontier)
	return None, len(explored)

def breadth_first_graph_search(_problem: GraphProblem, _debug) -> tuple[Node | None, int]:
	"""Checks all nodes at a depth before moving to the next depth,
	Note that this function can be implemented in a
	single line as below:
	return graph_search(problem, FIFOQueue())
	From AIMA.
	"""
	node = Node(_problem.initial)
	if _problem.goal_test(node.state):
		return node, int(1)
	frontier = deque([node])
	explored = set()
	while frontier:
		node = frontier.popleft()
		if _debug:
			print(node, node.expand(_problem))
		explored.add(node.state)
		for child in node.expand(_problem):
			if child.state not in explored and child not in frontier:
				if _problem.goal_test(child.state):
					return child, len(explored)
				frontier.append(child)
	return None, len(explored)

def best_first_graph_search(_problem: GraphProblem, _debug, _f, _display=False) -> tuple[Node | None, int]:
	"""Search the nodes with the lowest f scores first.
	You specify the function f(node) that you want to minimize; for example,
	if f is a heuristic estimate to the goal, then we have greedy best
	first search; if f is node.depth then we have breadth-first search.
	There is a subtlety: the line "f = memoize(f, 'f')" means that the f
	values will be cached on the nodes as they are computed. So after doing
	a best first search you can examine the f values of the path returned. From AIMA."""

	#f = memoize(f, 'f')
	_f = memoize(_f)
	node = Node(_problem.initial)
	frontier = PriorityQueue('min', _f)
	frontier.append(node)
	explored = set()
	while frontier:
		node = frontier.pop()
		if _debug:
			print(node, node.expand(_problem))
		if _problem.goal_test(node.state):
			if _display:
				print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
			return node, len(explored)
		explored.add(node.state)
		for child in node.expand(_problem):
			if child.state not in explored and child not in frontier:
				frontier.append(child)
			elif child in frontier:
				if _f(child) < frontier[child]:
					del frontier[child]
					frontier.append(child)
	return None, len(explored)

def uniform_cost_search(_problem: GraphProblem, _debug, _display=False) -> tuple[Node | None, int]:
	"""This is greedy best first search. From AIMA."""

	return best_first_graph_search(_problem, _debug, lambda node: node.path_cost, _display)

def depth_limited_search(_problem: GraphProblem, _debug, _limit=50):
	"""Uses recursion to explore nodes. From AIMA."""

	def recursive_dls(_node: Node, _problem: GraphProblem, _limit: int, _debug: bool):
		if _debug:
			print(_node, _node.expand(_problem))
		minLimit = _limit
		if _problem.goal_test(_node.state):
			return _node, minLimit
		elif _limit == 0:
			return 'cutoff', minLimit
		else:
			cutoff_occurred = False
			for child in _node.expand(_problem):
					result, minLimit = recursive_dls(child, _problem, _limit - 1, _debug)
					if result == 'cutoff':
						cutoff_occurred = True
					elif result is not None:
						return result, minLimit
			# If none is returned it means no path can be found
			if cutoff_occurred:
				return 'cutoff', minLimit
			else:
				return None, minLimit

	# Body of depth_limited_search:
	return recursive_dls(Node(_problem.initial), _problem, _limit, _debug)

def iterative_deepening_search(_problem: GraphProblem, _debug):
	"""Repeatedly calls depth limited search with more depth values. From AIMA."""

	total = 0
	for depth in range(sys.maxsize):
		result, limit = depth_limited_search(_problem, _debug, depth)
		# If limit is zero, the number of nodes checked is the depth
		total += depth - limit
		if result != 'cutoff':
			return result, total

	# Stupid fallback to shut python up
	return None, int(0)

def astar_search(_problem: GraphProblem, _debug, _h=None, _display=False) -> tuple[Node | None, int]:
	"""A* search is best-first graph search with f(n) = g(n)+h(n).
	You need to specify the h function when you call astar_search, or
	else in your Problem subclass. From AIMA."""

	#h = memoize(h or problem.h, 'h')
	_h = memoize(_h or _problem.h)
	return best_first_graph_search(_problem, _debug, lambda n: n.path_cost + _h(n), _display)

def beam_search(_problem: GraphProblem, _debug, _k=2) -> tuple[Node | None, int]:
	"""Works like a focused greedy best first search."""

	frontier = PriorityQueue('min', lambda n: n.path_cost)
	frontier.append(Node(_problem.initial))
	explored = set()
	while frontier:
		if len(frontier) > _k:
			frontier.heap = sorted(frontier.heap)[:_k]
		node = frontier.pop()
		if _debug:
			print(node, node.expand(_problem))
		if _problem.goal_test(node.state):
			return node, len(explored)
		explored.add(node.state)
		children = list(node.expand(_problem))
		new_candidates = [child for child in children if child.state not in explored]
		frontier.extend(new_candidates)
	return None, len(explored)

def import_graph(_file, _useChar = False, _debug=False):
	"""Import the graph data. Create the GraphProblem and return it, also return the goal."""

	def parse_graph(_file):
		"""Parse the graph file and import the raw data."""

		# Import the text file into a variable
		with open(_file, 'r') as _file:
			lines = _file.readlines()

		nodes: dict[int, tuple[int, int]] = {}
		edges: dict[int, dict[int, int]] = {}
		origin: int = -1
		destinations: list[int] = []
		section: str | None = None

		for line in lines:
			# Remove whitespaces
			line = line.strip()
			# Skip empty lines
			if not line:
				continue

			# Switch the section
			if line.startswith("Nodes:"):
				section = "nodes"
				continue
			elif line.startswith("Edges:"):
				section = "edges"
				continue
			elif line.startswith("Origin:"):
				section = "origin"
				continue
			elif line.startswith("Destinations:"):
				section = "destinations"
				continue

			match section:
				# Import the nodes
				case "nodes":
					# Match with regex
					match = re.match(r'(\d+): \((\d+),(\d+)\)', line)
					if match:
						# Extract the node and the coordinates
						node, x, y = match.groups()
						# Store the nodes using the number ID, duplicating an ID will overwrite it.
						nodes[int(node)] = (int(x), int(y))

				# Import the edges
				case "edges":
					# Match with regex
					match = re.match(r'\((\d+),(\d+)\): (\d+)', line)
					if match:
						# Extract the start node, end node and goal
						n1, n2, cost = match.groups()
						n1, n2, cost = int(n1), int(n2), int(cost)
						# Create a dictionary element for the first node if it does not yet exist
						if n1 not in edges:
							edges[n1] = {}
						edges[n1][n2] = cost

				case "origin":
					origin = int(line)

				case "destinations":
					destinations = [int(val) for val in line.split(';')]

		# Sorting the dictionaries, dunno if this is needed but I like it
		nodes = dict(sorted(nodes.items()))
		edges = dict(sorted(edges.items()))
		return nodes, edges, origin, destinations

	def create_graph_problem(
		_nodes: dict[int, tuple[int, int]],
		_edges: dict[int, dict[int, int]],
		_origin: int,
		_destinations: list[int],
		_useChar: bool,
		_debug: bool
	):
		"""Take the raw data from the graph file and convert it into a graph problem."""

		edges = {}
		locations = {}
		origin = 0
		goals = []

		# Old code, converted the numeric id into a character, 64 being the offset before A
		if _useChar:
			edges = {chr(64 + key): {chr(64 + k): v for k, v in value.items()} for key, value in _edges.items()}
			locations = {chr(64 + key): value for key, value in _nodes.items()}
			origin = chr(64 + _origin)
			goals = [chr(64 + value) for value in _destinations]
		else:
			edges = {key: {k: v for k, v in value.items()} for key, value in _edges.items()}
			locations = {key: value for key, value in _nodes.items()}
			origin = _origin
			goals = _destinations

		# Useful debugging information
		if _debug:
			print("Edges:", edges, sep=" ")
			print("Locations:", locations, sep=" ")
			print("Origin:", origin, sep=" ")
			print("Destinations:", goals, sep=" ")

		graph = Graph(edges)
		graph.locations = locations
		return GraphProblem(origin, goals, graph), goals

	nodes, edges, origin, destinations = parse_graph(_file)
	problem, goals = create_graph_problem(nodes, edges, origin, destinations, _useChar, _debug)
	return problem, goals

def select_method(_method: str):
	"""Return a pathfinding function."""
	match _method:
		case "DFS":
			return depth_first_graph_search
		case "BFS":
			return breadth_first_graph_search
		case "GBFS":
			return uniform_cost_search
		case "AS":
			return astar_search
		case "IDS" | "CUS1":
			return iterative_deepening_search
		case "BS" | "CUS2":
			return beam_search
		case _:
			return None

def main(_output = True, _debug = False):
	"""Main execution of the script. Returns the exection time."""

	# Not enough arguments
	if len(sys.argv) < 3:
		print("Missing arguments: python search.py <filename> <method>")
		quit()

	# Too many arguments
	if len(sys.argv) > 3:
		print("Excess arguments: python search.py <filename> <method>")

	# Import the graph file
	graph_problem, goals = import_graph(sys.argv[1], False, _debug)

	# Extract parameter 2: "method" function used
	method = select_method(sys.argv[2])

	if method is None:
		print("Incorrect method type, valid methods:\nDFS, BFS, GBFS, AS, CUS1, CUS2, IDS, BS")
		quit()

	start_time = time.time()
	# Call the selected pathfinding method
	result, count = method(graph_problem, _debug)
	duration = time.time() - start_time

	if _debug:
		print("Completed pathfinding in ", duration)

	if _output:
		# Output paramter 1
		print("filename=", sys.argv[1], sep="", end=" | ")
		# Output paramter 2
		print("method=", sys.argv[2], sep="")
		# \n
		# Ouput goal node
		print("goal=", goals, sep="", end=" | ")

		# Output number (length of path)
		print("number of nodes=", count, sep="")
		# \n
		if (result is not None):
			# Output path: list of nodes
			print("path=", result.solution(), sep="")
		else:
			print("No path found!")

	return duration

def compute_average_runtime(_freq = 25000):
	"""Exectues the main function repeatedly"""

	sum = 0
	for num in range(_freq):
		sum += main(False)

	print("Average exection time accross ", _freq, " runs is ", sum / _freq)

if __name__ == "__main__":
	main()
	#compute_average_runtime()
