import plotly.express as px

# 1. Filter for a specific region and year
df = px.data.gapminder().query("year == 2007 and continent == 'Europe'")

# 2. Build the bar chart
fig = px.bar(
    df, 
    x='country', 
    y='gdpPercap',
    color='gdpPercap',           # Use a color gradient for extra clarity
    color_continuous_scale='Viridis',
    text_auto='.3s',              # Add values directly on the bars
    title="Europe: GDP per Capita Comparison (2007)",
    labels={'gdpPercap': 'GDP per Capita (USD)', 'country': 'Country'}
)

# 3. The "Intuitive" Upgrade: Sorting & Layout
# 'total descending' makes the tallest bars appear on the left automatically
fig.update_layout(
    xaxis={'categoryorder':'total descending'},
    template="plotly_white",
    coloraxis_showscale=False # Keeps the look clean
)

fig.show()