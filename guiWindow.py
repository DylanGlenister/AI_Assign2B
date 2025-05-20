import tkinter as tk
from tkinter import ttk
import pandas as pd
import subprocess
import webbrowser
import os

# Load SCATS reference list
scats_df = pd.read_csv("scats_reference.csv")
scat_sites = sorted(scats_df['Site_Number'].unique().astype(str))

# Create main window
root = tk.Tk()
root.title("Traffic-Based Route Guidance System")
root.geometry("500x600")

# Main Frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Input Frame
input_frame = tk.Frame(main_frame)
input_frame.pack(pady=20, padx=20)

# Routes Frame
routes_frame = tk.LabelFrame(main_frame, text="Routes", padx=10, pady=10)
routes_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

# Input Fields with Dropdowns
fields = ["Start Scat", "End Scat", "Start time", "Model"]
entries = {}

for field in fields:
    row = tk.Frame(input_frame)
    row.pack(fill=tk.X, pady=5)

    label = tk.Label(row, width=15, text=field, anchor='w')
    label.pack(side=tk.LEFT)

    if field in ["Start Scat", "End Scat"]:
        combo = ttk.Combobox(row, values=scat_sites)
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entries[field] = combo
    elif field == "Model":
        combo = ttk.Combobox(row, values=["LSTM", "GRU", "Other"])
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entries[field] = combo
    else:
        entry = tk.Entry(row)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entries[field] = entry

# Route Display Placeholders
for i in range(5):
    tk.Label(routes_frame, text=f"Route {i+1}: [Details here]").pack(anchor='w')

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

# Start GUI
root.mainloop()
