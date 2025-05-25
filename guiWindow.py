import os
import tkinter as tk
import webbrowser
from tkinter import ttk

import pandas as pd

import generateMap
import route_finder
import shared


# Create main window
root = tk.Tk()
root.title('Traffic-Based Route Guidance System')
root.geometry('800x600')

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

def load_SCATs() -> list[str]:
	'''Reads in the dataset and imports all the unique SCATs sites.'''
	# Load SCATS reference list
	scats_df = pd.read_csv(shared.PATH_DATASET)
	scats_sorted = pd.Series(sorted(scats_df[shared.COLUMN_SCAT].unique()))
	return scats_sorted.astype(str).tolist()

SCATs = load_SCATs()

# Input Fields with Dropdowns
def create_field(_name: str, _values: list[str]):
	'''Creat an input field.'''
	row = tk.Frame(input_frame)
	row.pack(fill=tk.X, pady=5)
	label = tk.Label(row, width=15, text=_name, anchor='w')
	label.pack(side=tk.LEFT)
	combo = ttk.Combobox(row, values=_values)
	combo.current(0)
	combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
	entries[_name] = combo

create_field('Start SCAT', SCATs)
create_field('End SCAT', SCATs)
create_field('Start Time', timeset)
create_field('Model', models)

# === Routes Display Box ===
routes_frame = tk.LabelFrame(main_frame, text='Routes', padx=10, pady=10)
routes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

# Add placeholder labels
route_labels: list[tk.Label] = []
for i in range(5):
	lbl = tk.Label(routes_frame, text=f'')
	lbl.pack(anchor='w')
	route_labels.append(lbl)

def try_open_map(_path: str):
	try:
		map_path = os.path.abspath(_path)
		webbrowser.open(f'file://{map_path}')
	except Exception as e:
		print('Error generating or opening map:', e)

# === Generate Map Button ===
def calculate_routes():

	# Clear all the labels
	for i in range(5):
		route_labels[i].config(text=f'')

	origin = entries['Start SCAT'].get()
	goal = entries['End SCAT'].get()
	time = entries['Start Time'].get()
	model = entries['Model'].get()

	# Input error checking
	if origin == '':
		route_labels[0].config(text=f'No start SCAT set!')
		return
	if goal == '':
		route_labels[0].config(text=f'No end SCAT set!')
		return
	if origin == goal:
		route_labels[0].config(text=f'Start and end SCATs cannot be the same!')
		return

	# Convert SCATs sites to ints
	origin = int(origin)
	goal = int(goal)

	# Will find 5 paths, return value will be a list
	paths = route_finder.find_five_routes( origin, goal, time, model)

	route_colours = [
		('red', '#FF0000'),
		('purple', "#AA4CE9"),
		('green', '#32C032'),
		('blue', '#2F3CF0'),
		('cyan', '#5CB9F7')
	]

	# Use enumerate to get an index
	for i, path in enumerate(paths):
		if path is None:
			# No path found, alert user
			route_labels[0].config(text=f'No path found!')

		# Display route to user
		colour_label, _ = route_colours[i]
		route_labels[i].config(text=f'Route {i+1} ({colour_label}): {path}')

	generateMap.generate_map(origin, goal, paths, route_colours)
	try_open_map(shared.PATH_ROUTEMAP)

def display_graph():
	generateMap.graph_visualisation()
	try_open_map(shared.PATH_GRAPHMAP)

map_btn = tk.Button(main_frame, text='Calculate route', command=calculate_routes)
map_btn.pack(pady=10)

graph_btn = tk.Button(main_frame, text='Display graph', command=display_graph)
graph_btn.pack(pady=10)

# Start the GUI
root.mainloop()
