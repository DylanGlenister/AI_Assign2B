import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
import subprocess
import webbrowser

# Load processed data
df = pd.read_csv("processed.csv")

# Dropdown values
scat_sites = sorted(df['SCATS'].dropna().unique().astype(str))
time_columns = df.columns[8:]  # Skip SCATS and Day_of_week
time_list = sorted(list(time_columns))  # Ensure times are in order

# Create main window
root = tk.Tk()
root.title("Traffic-Based Route Guidance System")
root.geometry("500x600")

# Main Frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Input Frame
input_frame = tk.Frame(main_frame)
input_frame.pack(pady=10)

# Input Fields
fields = ["Start Scat", "End Scat", "Start Time", "Model"]
entries = {}

for field in fields:
    row = tk.Frame(input_frame)
    row.pack(fill=tk.X, pady=5)

    label = tk.Label(row, width=15, text=field, anchor='w')
    label.pack(side=tk.LEFT)

    # Set values based on field type
    if field in ["Start Scat", "End Scat"]:
        combo = ttk.Combobox(row, values=scat_sites)
    elif field == "Start Time":
        combo = ttk.Combobox(row, values=time_list)
    elif field == "Model":
        combo = ttk.Combobox(row, values=["LSTM", "GRU", "Other"])
    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
    entries[field] = combo

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
def generate_map():
    try:
        subprocess.run(["python", "generateMap.py"], check=True)
        map_path = os.path.abspath("map.html")
        webbrowser.open(f"file://{map_path}")
    except Exception as e:
        print("Error generating or opening map:", e)

map_btn = tk.Button(main_frame, text="Generate Map", command=generate_map)
map_btn.pack(pady=10)

# Start the GUI
root.mainloop()
