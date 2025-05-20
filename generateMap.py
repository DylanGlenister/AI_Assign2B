import folium as fm


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

	# NOTE This does not factor the offsets, which need to be applied in process_data.py

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
