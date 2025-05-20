import os
import tkinter as tk
import webbrowser
from tkinter import ttk

import pandas as pd

import construct_graph
import generateMap
import search

# Load SCATS reference list
scats_df = pd.read_csv('./processed.csv')
scats_sorted = pd.Series(sorted(scats_df['SCATS'].unique()))
scat_sites = scats_sorted.astype(str).tolist()

# Create main window
root = tk.Tk()
root.title('Traffic-Based Route Guidance System')
root.geometry('500x600')

# === Main Frame ===
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# === Input Frame ===
input_frame = tk.Frame(main_frame)
input_frame.pack(pady=10)

# Create an array of times that can be selected from
timeset: list[str] = []
for hour in range(24):
	for minute in range(0, 46, 15):
		timeset.append(f'{hour:0=2}:{minute:0=2}')

# A list of the available models
models = ['LSTM', 'GRU', 'Other']

# References to each of the input boxes
entries: dict[str, ttk.Combobox] = {}

# Input Fields with Dropdowns
def create_field(_name: str, _values: list[str]):
	row = tk.Frame(input_frame)
	row.pack(fill=tk.X, pady=5)
	label = tk.Label(row, width=15, text=_name, anchor='w')
	label.pack(side=tk.LEFT)
	combo = ttk.Combobox(row, values=_values)
	combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
	entries[_name] = combo

create_field('Start SCAT', scat_sites)
create_field('End SCAT', scat_sites)
create_field('Start Time', timeset)
create_field('Models', models)

# === Routes Frame ===
routes_frame = tk.LabelFrame(main_frame, text='Routes', padx=10, pady=10)
routes_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

# === Placeholder Button ===
def placeholder_action():
	print("This will trigger the ML model and show the routes.")
	# For now, update route labels with placeholder text
	for i in range(5):
		route_labels[i].config(text=f"Route {i+1}: [Generated route {i+1} shown here]")

placeholder_btn = tk.Button(main_frame, text="Calculate Route", command=placeholder_action)
placeholder_btn.pack(pady=10)

# === Routes Display Box ===
routes_frame = tk.LabelFrame(main_frame, text="Routes", padx=10, pady=10)
routes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# Add placeholder labels
route_labels = []
for i in range(5):
	lbl = tk.Label(routes_frame, text=f"Route {i+1}: [Details here]")
	lbl.pack(anchor='w')
	route_labels.append(lbl)

# === Generate Map Button ===
def calculate_route():

	origin = entries['Start SCAT'].get()
	goal = entries['End SCAT'].get()
	#entries['Start Time'].get(),
	#entries['Model'].get()

	if origin == '':
		# No origin set, alert user
		tk.Label(routes_frame, text='No start SCAT set!').pack(anchor='w')
		return

	if goal == '':
		# No goal set, alert user
		tk.Label(routes_frame, text='No end SCAT set!').pack(anchor='w')
		return

	origin = int(origin)
	goal = int(goal)

	# Need to pass information from model into this
	graph, locations = construct_graph.create_graph(scats_df)

	problem = search.GraphProblem(origin, goal, graph)

	# Result is search.Node and count is int
	result, count = search.astar_search(problem, False)

	if result is None:
		# No path found, alert user
		tk.Label(routes_frame, text='No path found!').pack(anchor='w')
		return

	path: list[search.Node] = result.solution()

	tk.Label(routes_frame, text=f'Route: {path}').pack(anchor='w')

	generateMap.generate_map(origin, goal, path, locations)

	try:
		map_path = os.path.abspath('map.html')
		webbrowser.open(f'file://{map_path}')
	except Exception as e:
		print('Error generating or opening map:', e)

map_btn = tk.Button(main_frame, text='Calculate route', command=calculate_route)
map_btn.pack(pady=10)

# Start the GUI
root.mainloop()
