import folium as fm

# Change to a list of 5 tuple with start, end, path, colour
def generate_map(_start: int, _end: int, _path, _locations: dict[int, tuple[float, float]]):
	start_pos = _locations[_start]
	prev = start_pos
	m = fm.Map(location=start_pos, zoom_start = 15)
	fm.Marker(start_pos).add_to(m)

	for node in _path:
		position = _locations[node]
		fm.PolyLine((prev, position)).add_to(m)
		prev = position

	fm.Marker(_locations[_end]).add_to(m)

	m.save('map.html')

def show_all_nodes(_raw_graph: tuple[dict[int, dict[int, int]], dict[int, tuple[float, float]]]):
	m = fm.Map(location=(-37.80486 + 0.00142, 145.08093 + 0.00171), zoom_start = 15)

	edges, locations = _raw_graph

	# List all edges and their cost
	for node, others in edges.items():
		fm.Marker(locations[node]).add_to(m)
		for other, cost in others.items():
			fm.PolyLine((locations[node], locations[other])).add_to(m)

	m.save('graph.html')

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
	m.save('map.html')

if __name__ == '__main__':
	run_test()
