from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from sandbox import run_sandboxed

rows = [
    {"road_type": "highway", "accident_severity": "fatal", "accident_count": 983},
    {"road_type": "highway", "accident_severity": "major", "accident_count": 1967},
    {"road_type": "highway", "accident_severity": "minor", "accident_count": 3666},
    {"road_type": "rural", "accident_severity": "fatal", "accident_count": 965},
    {"road_type": "rural", "accident_severity": "major", "accident_count": 2014},
    {"road_type": "rural", "accident_severity": "minor", "accident_count": 3660},
    {"road_type": "urban", "accident_severity": "fatal", "accident_count": 1039},
    {"road_type": "urban", "accident_severity": "major", "accident_count": 2007},
    {"road_type": "urban", "accident_severity": "minor", "accident_count": 3699},
]

code = """
matrix_data = df.groupby(['road_type', 'accident_severity']).size().reset_index(name='accident_count')
pivot_df = matrix_data.pivot(index='road_type', columns='accident_severity', values='accident_count').fillna(0)

fig = px.imshow(
    pivot_df,
    text_auto=True,
    aspect='auto',
    template='plotly_dark',
    title='Road Type vs Accident Severity Matrix',
    labels=dict(x='Accident Severity', y='Road Type', color='Accident Count'),
    color_continuous_scale='Viridis'
)

log_output(matrix_data)
result_df = df
"""

result = run_sandboxed(code, rows)
print("query_output:")
print(result.query_output)
print("\nvisualization keys:", list((result.visualization or {}).keys()))
trace = (result.visualization or {}).get("data", [{}])[0]
print("trace type:", trace.get("type"))
z = trace.get("z")
print("z type:", type(z).__name__)
print("z value:", z)
print("x:", trace.get("x"))
print("y:", trace.get("y"))
