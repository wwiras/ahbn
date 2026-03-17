import plotly.express as px

# 1. Load the built-in dataset
df = px.data.gapminder()

# 2. Build the interactive bubble chart
fig = px.scatter(
    df,
    x="gdpPercap",          # X-axis: Wealth
    y="lifeExp",            # Y-axis: Health
    animation_frame="year", # Adds the time slider
    animation_group="country",
    size="pop",             # Bubble size: Population
    color="continent",      # Bubble color: Continent
    hover_name="country",   # Labels on hover
    log_x=True,             # Log scale for better visibility of values
    size_max=55,
    range_x=[100, 100000],
    range_y=[25, 90],
    title="Global Development: Wealth vs. Health (1952-2007)",
    labels={"gdpPercap": "GDP per Capita", "lifeExp": "Life Expectancy"}
)

# 3. Clean up the styling
fig.update_layout(template="plotly_white")

# 4. Display the chart
fig.show()