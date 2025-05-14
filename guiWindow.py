import tkinter as tk
from tkinter import ttk
import pandas as pd
import webview

# Load SCATS reference list
scats_df = pd.read_csv("scats_reference.csv")

# Extract and sort SCAT site numbers
scat_sites = sorted(scats_df['Site_Number'].unique().astype(str))  # Convert to string for tkinter

# Create main window
root = tk.Tk()
root.title("Traffic-Based Route Guidance System")
root.geometry("1000x600")

# Left Frame
left_frame = tk.Frame(root, width=400, height=600)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# Input Frame
input_frame = tk.Frame(left_frame)
input_frame.pack(pady=20, padx=20)

# Routes Frame
routes_frame = tk.LabelFrame(left_frame, text="Routes", padx=10, pady=10)
routes_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

# Map Frame
map_frame = tk.LabelFrame(root, text="Map", width=600, height=600)
map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

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

with open('./map.html', 'r', encoding='utf-8') as mapfile:
    maphtml = mapfile.read()

# Map Placeholder
#map_placeholder = tk.Label(map_frame, text="Map Display Area", bg="white")
#map_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

webview.create_window("Fuck", html="./map.html")
webview.start()

# Start GUI
root.mainloop()
