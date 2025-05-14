import folium as fm
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
