from __future__ import annotations

import pandas as pd
import plotly.express as px

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

df = pd.DataFrame(rows)
pivot_df = df.pivot(index='road_type', columns='accident_severity', values='accident_count').fillna(0)

fig = px.imshow(
    pivot_df,
    text_auto=True,
    aspect='auto',
    template='plotly_dark',
    title='Road Type vs Accident Severity Matrix',
    labels=dict(x='Accident Severity', y='Road Type', color='Accident Count'),
    color_continuous_scale='Viridis'
)

j = fig.to_plotly_json()
z = j['data'][0]['z']
print(type(z).__name__)
print(z)
print('keys:' if isinstance(z, dict) else 'len:', list(z.keys()) if isinstance(z, dict) else len(z))
