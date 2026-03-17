import plotly.express as px

# 1. Load data and filter for 5 major economies (to avoid a "spaghetti" mess)
df = px.data.gapminder()
nations = ["United States", "China", "India", "Japan", "United Kingdom"]
df_filtered = df[df['country'].isin(nations)]

# 2. Build the multi-line chart
fig = px.line(
    df_filtered, 
    x="year", 
    y="gdpPercap", 
    color="country",
    markers=True, # Adding markers makes data points easier to pinpoint
    title="Economic Growth Comparison: GDP per Capita (1952–2007)",
    labels={"year": "Year", "gdpPercap": "GDP per Capita (USD)"}
)

# 3. The "Intuition" Upgrade: Unified Hover
# This shows all country values for a specific year in one tooltip.
fig.update_layout(
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(showgrid=False) # Removes vertical lines to focus on the trend
)

fig.show()