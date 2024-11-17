#!/usr/bin/env python
# coding: utf-8

# # Economic Data Dashboard Notebook
# 

# # http://127.0.0.1:8042/

# # 1. Configuration and Initialization. (FredApi and Start Date)

# In[1]:


# Configuration and Initialization
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta




# FRED API Configuration
api_key = '7227512392e5e5d2a2679a261d2bb3a9'
fred = Fred(api_key=api_key)

# Define the start and end dates
start = '2015-01-01'
end = '2024-07-01'


# In[19]:


import investpy
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get economic calendar data for the United States from the year 2020 onwards
economic_calendar_data = investpy.economic_calendar(
    countries=['united states'],
    from_date='01/01/2017',
    to_date='31/12/2024'
)

# Convert the data to a DataFrame
calendar_df = pd.DataFrame(economic_calendar_data)


# # 2. Fetching Data from FRED

# # CPI Page

# ## Graphs

# In[2]:


def generate_cpi_detailed_graph_1():
    # Initialize FRED with the provided API Key
    fred = Fred(api_key=api_key)
    
    # Retrieve CPI and Core CPI data
    cpi_data = fred.get_series('CPIAUCSL')
    core_cpi_data = fred.get_series('CPILFESL')
    
    # Resample data monthly and compute the six-month annualized growth rate
    def annualized_growth_rate(series, window=6):
        return (series.pct_change(window) + 1) ** (12/window) - 1
    
    cpi_growth = annualized_growth_rate(cpi_data)
    core_cpi_growth = annualized_growth_rate(core_cpi_data)
    
    # Compute YoY percentage change for both series
    cpi_yoy = cpi_data.pct_change(periods=12) * 100
    core_cpi_yoy = core_cpi_data.pct_change(periods=12) * 100
    
    # Filter data from 2015 onwards
    cpi_growth = cpi_growth[cpi_growth.index >= '2015-01-01']
    core_cpi_growth = core_cpi_growth[core_cpi_growth.index >= '2015-01-01']
    cpi_yoy = cpi_yoy[cpi_yoy.index >= '2015-01-01']
    core_cpi_yoy = core_cpi_yoy[core_cpi_yoy.index >= '2015-01-01']
    
    # Prepare date range for plotting
    dates = cpi_growth.index
    
    # Get the most recent readings for both series
    recent_cpi_value = cpi_growth.dropna().iloc[-1] * 100
    recent_core_cpi_value = core_cpi_growth.dropna().iloc[-1] * 100
    recent_cpi_yoy_value = cpi_yoy.dropna().iloc[-1]
    recent_core_cpi_yoy_value = core_cpi_yoy.dropna().iloc[-1]
    
    # Prepare data for the table with new order and labels
    table_data = pd.DataFrame({
        "CPI YoY (%)": cpi_yoy.dropna().tail(30).values,
        "CPI 6M (%)": cpi_growth.dropna().tail(30).values * 100,
        "Core CPI YoY (%)": core_cpi_yoy.dropna().tail(30).values,
        "Core CPI 6M (%)": core_cpi_growth.dropna().tail(30).values * 100
    }).T.round(1)
    
    # Extract the dates as column headers
    dates_for_table = cpi_growth.dropna().tail(30).index.strftime("%y-%m")
    
    # Heatmap color functions
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'
    
    # Generate colors for the table based on the data rows
    colors = [get_row_heatmap_colors(row) for _, row in table_data.iterrows()]
    
    # Transpose the list of colors to match Plotly's requirement (each sublist represents a column of colors)
    flat_colors = list(map(list, zip(*colors)))
    
    # Create subplots: 2 columns for charts and 1 row for the table
    fig = make_subplots(
        rows=2, cols=2, 
        row_heights=[0.6, 0.4],  # 60% for charts, 40% for table
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "table", "colspan": 2}, None]],  # Use full width for the table
        subplot_titles=('Headline CPI', 'Core CPI'),
        vertical_spacing=0.05  # Smaller gap between plots and table
    )
    
    # Add Headline CPI chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=cpi_growth * 100, mode='lines', name='6M Annualized CPI', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_cpi_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=1)
    
    # Add Core CPI chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=core_cpi_growth * 100, mode='lines', name='6M Annualized Core CPI', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_core_cpi_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=2)
    
    # Add YoY CPI (White)
    fig.add_trace(go.Scatter(
        x=dates, y=cpi_yoy, mode='lines', name='YoY CPI', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=1)
    
    # Add YoY Core CPI (White)
    fig.add_trace(go.Scatter(
        x=dates, y=core_cpi_yoy, mode='lines', name='YoY Core CPI', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=2)
    
    # Add a single 2% Fed target line to both charts (only one shown in legend)
    fig.add_trace(go.Scatter(
        x=dates, y=[2]*len(dates), mode='lines', name='Fed Reserve Goal', 
        line=dict(color='crimson', dash='dash')
    ), row=1, col=1)
    
    # Duplicate the Fed target line for the second chart without showing it in the legend
    fig.add_trace(go.Scatter(
        x=dates, y=[2]*len(dates), mode='lines', name='', showlegend=False, 
        line=dict(color='crimson', dash='dash')
    ), row=1, col=2)
    
    # Add the initial table below the charts
    table = go.Table(
        header=dict(
            values=["Items"] + list(dates_for_table),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_data.index,  # Row labels
                *[table_data[col] for col in table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(dates_for_table)] + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        ),
        columnwidth=[4] + [0.5] * len(dates_for_table)  # Adjusted column widths
    )
    fig.add_trace(table, row=2, col=1)  # Table spans both columns
    
    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added
    
    # Calculate global y-axis range for both plots
    # Combine all y-values from lines and bars to determine global min and max
    y_values = []
    for column in ['cpi_growth', 'core_cpi_growth', 'cpi_yoy', 'core_cpi_yoy']:
        if column == 'cpi_growth':
            y_values.extend(cpi_growth.dropna().values * 100)
        elif column == 'core_cpi_growth':
            y_values.extend(core_cpi_growth.dropna().values * 100)
        elif column == 'cpi_yoy':
            y_values.extend(cpi_yoy.dropna().values)
        elif column == 'core_cpi_yoy':
            y_values.extend(core_cpi_yoy.dropna().values)
    global_y_min = min(y_values)
    global_y_max = max(y_values)
    # Add some padding
    padding = (global_y_max - global_y_min) * 0.05
    global_y_min -= padding
    global_y_max += padding
    
    # Define rolling windows (e.g., 30-month rolling windows)
    window_size = 30  # Number of months in each window
    windows = []
    for i in range(len(cpi_growth) - window_size + 1):
        window = {
            'cpi_yoy': cpi_yoy.iloc[i:i+window_size],
            'cpi_growth': cpi_growth.iloc[i:i+window_size],
            'core_cpi_yoy': core_cpi_yoy.iloc[i:i+window_size],
            'core_cpi_growth': core_cpi_growth.iloc[i:i+window_size],
            'dates': cpi_growth.index[i:i+window_size]
        }
        windows.append(window)
    
    # Function to prepare table data for a given window
    def prepare_table_data(window):
        window_data = {
            "CPI YoY (%)": window['cpi_yoy'].values,
            "CPI 6M (%)": window['cpi_growth'].values * 100,
            "Core CPI YoY (%)": window['core_cpi_yoy'].values,
            "Core CPI 6M (%)": window['core_cpi_growth'].values * 100
        }
        table_df = pd.DataFrame(window_data).T.round(1)
        window_dates = window['dates'].strftime("%y-%m")
        return table_df, window_dates
    
    # Generate frames for each window
    frames = []
    for i, window in enumerate(windows):
        table_df, window_dates = prepare_table_data(window)
        
        # Generate colors for the table
        window_colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
        window_flat_colors = list(map(list, zip(*window_colors)))  # Transpose
        
        # Define updated table cells
        updated_cells = dict(
            values=[
                table_df.index,  # Row labels
                *[table_df[col] for col in table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(window_dates)] + window_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        )
        
        # Define updated table header
        updated_header = dict(
            values=["Items"] + list(window_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )
        
        # Calculate the start and end dates for the shaded window
        window_start = window['dates'].min()
        window_end = window['dates'].max()
        
        # Define the shaded rectangle shapes for both plots
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        shaded_shape_plot2 = dict(
            type="rect",
            xref="x2",  # References the x-axis of the second subplot
            yref="y2",  # References the y-axis of the second subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        # Create the frame with updated table and shaded shapes
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1, shaded_shape_plot2]  # Add shaded areas to both plots
            ),
            name=str(i),
            traces=[table_trace_index]  # Update the existing table trace
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Define slider steps
    slider_steps = []
    for i in range(len(windows)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows[i]['dates'].min().strftime("%Y-%m")  # Label represents the start date of the window
        )
        slider_steps.append(step)
    
    # Optionally, reverse the slider steps to have newest on the left
    slider_steps = slider_steps[::-1]
    
    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]
    
    # Add the slider to the layout
    fig.update_layout(
        sliders=sliders
    )
    
    # Add annotations for the latest values on both plots
    fig.add_annotation(
        x=dates[-1], y=recent_cpi_value, xref="x1", yref="y1", 
        text=f"Latest: {recent_cpi_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    fig.add_annotation(
        x=dates[-1], y=recent_core_cpi_value, xref="x2", yref="y2", 
        text=f"Latest: {recent_core_cpi_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    # Add source annotation at the bottom of the chart with hyperlinks
    fig.add_annotation(
        text=('Source: Federal Reserve Economic Data (FRED). '
              '<a href="https://fred.stlouisfed.org/graph/?g=1tEZX" style="color: white">Headline CPI</a>, '
              '<a href="https://fred.stlouisfed.org/graph/?g=1tF03" style="color: white">Core CPI</a>.'),
        font=dict(size=8, color="white"),  # Font size and color
        xref="paper", yref="paper",  # Use paper coordinates
        x=0.5, y=-0.03,  # Position the source annotation at the bottom
        showarrow=False
    )
    
    # Update layout and labels
    fig.update_layout(
        title="",
        yaxis_title="Growth Rate (%)",
        yaxis2_title="Growth Rate (%)",
        template="plotly_dark",
        font=dict(color="white", size=8),
        title_x=0.5,
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="top",  # Anchor the legend to the top
            y=1.2,  # Set the legend position slightly below the title
            xanchor="center",  # Center the legend
            x=0.5,
            font=dict(size=8)  # Set the font size for the legend
        ),
        height=600,  # Adjusted height
        width=1000,  # Adjusted width
        margin=dict(t=50, b=50, l=20, r=20)  # Adjusted side margins
    )
    
    # Fix the y-axis range for both plots to prevent dynamic scaling
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=1)
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=2)
    
    # Update Y-axis labels to reflect that we're displaying Contributions (percentage points)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
    
    return fig


# In[3]:


def generate_cpi_detailed_graph_2():
    # Initialize Fred API with your API key
    fred = Fred(api_key=api_key)  # Replace with your actual FRED API key

    # Define the FRED series IDs for the different CPI components, including Core Goods
    series_ids = {
        "All Items": "CPIAUCSL",               # All Items CPI
        "Core CPI": "CPILFESL",                # Core CPI
        "Food": "CPIUFDNS",                    # Food CPI
        "Energy": "CPIENGSL",                  # Energy CPI
        "Core Services": "CUSR0000SASLE",      # Core Services CPI
        "Core Goods": "CUSR0000SACL1E"         # Core Goods CPI
    }

    # Fetch the data from FRED
    cpi_data = {name: fred.get_series(series_id) for name, series_id in series_ids.items()}

    # Convert the data into a DataFrame
    cpi_df = pd.DataFrame(cpi_data)

    # Ensure the DataFrame index is in datetime format
    cpi_df.index = pd.to_datetime(cpi_df.index)

    # Define the display start date
    display_start_date = datetime(2015, 1, 1)

    # Calculate MoM data
    cpi_df_mom = cpi_df.pct_change(periods=1) * 100  # MoM percentage change

    # Calculate additional statistics for MoM
    cpi_df_mom['All Items (3-mo MA)'] = cpi_df_mom['All Items'].rolling(window=3).mean()
    cpi_df_mom['Core CPI (3-mo MA)'] = cpi_df_mom['Core CPI'].rolling(window=3).mean()

    # Drop rows with NaN values after calculations
    cpi_df_mom = cpi_df_mom.dropna()

    # Further filter data to display only from display_start_date onwards
    cpi_df_mom = cpi_df_mom[cpi_df_mom.index >= display_start_date]

    # Define window size
    window_size = 30  # Number of months in each window

    # Create subplots: 3 rows (2 for plots, 1 for table)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.35, 0.3],  # Adjust the heights of the rows
        subplot_titles=("Core CPI & All Items CPI %", "Selected Components % (Stacked)", "CPI Table"),
        vertical_spacing=0.05,  # Smaller gap between plots and the table
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )

    # Plot 1: Core CPI and All Items CPI (MoM)
    fig.add_trace(go.Scatter(
        x=cpi_df_mom.index, y=cpi_df_mom['All Items'],
        mode='lines', name='All Items CPI %',
        line=dict(color='white')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cpi_df_mom.index, y=cpi_df_mom['Core CPI'],
        mode='lines', name='Core CPI %',
        line=dict(color='magenta')
    ), row=1, col=1)

    # Add 3-month moving average (MoM)
    fig.add_trace(go.Scatter(
        x=cpi_df_mom.index, y=cpi_df_mom['All Items (3-mo MA)'],
        mode='lines', name='All Items (3-mo MA)',
        line=dict(color='white', dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cpi_df_mom.index, y=cpi_df_mom['Core CPI (3-mo MA)'],
        mode='lines', name='Core CPI (3-mo MA)',
        line=dict(color='magenta', dash='dot')
    ), row=1, col=1)

    # Plot 2: Stacked bar plot for components (MoM)
    fig.add_trace(go.Bar(
        x=cpi_df_mom.index, y=cpi_df_mom['Core Goods'], name='Core Goods %', marker_color='purple'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=cpi_df_mom.index, y=cpi_df_mom['Core Services'], name='Core Services %', marker_color='green'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=cpi_df_mom.index, y=cpi_df_mom['Energy'], name='Energy %', marker_color='orange'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=cpi_df_mom.index, y=cpi_df_mom['Food'], name='Food %', marker_color='blue'
    ), row=2, col=1)

    # Prepare data for the table (MoM)
    if len(cpi_df_mom) < window_size:
        raise ValueError(f"Not enough data points to create a window of size {window_size}.")

    # Generate windows for MoM
    windows_mom = [cpi_df_mom.iloc[i:i + window_size] for i in range(len(cpi_df_mom) - window_size + 1)]

    # Function to generate colors for each row based on values
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 - norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Prepare the initial table data (MoM)
    initial_window = windows_mom[0]
    table_df = initial_window[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].round(1)
    table_df = table_df.T  # Transpose for desired format
    table_header_dates = table_df.columns.strftime("%y-%m")

    # Generate colors for the initial table
    colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
    flat_colors = list(map(list, zip(*colors)))  # Transpose to match Plotly's requirement

    # Add the initial table
    table_trace = go.Table(
        header=dict(
            values=["Items"] + list(table_header_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_df.index,  # Row labels
                *[table_df[col] for col in table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(table_header_dates)] + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        ),
        columnwidth=[4] + [0.5] * len(table_header_dates)  # Adjust column widths
    )
    fig.add_trace(table_trace, row=3, col=1)  # Table is the third subplot

    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added

    # Function to prepare table data for a given window
    def prepare_table_data(window):
        window_data = window[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].round(1)
        window_df = window_data.copy()
        window_df = window_df.T  # Transpose for desired format
        window_dates = window_df.columns.strftime("%y-%m")
        return window_df, window_dates

    # Generate frames for the slider (MoM)
    frames_mom = []
    for i, window in enumerate(windows_mom):
        frame_table_df, window_dates = prepare_table_data(window)

        # Generate colors for the table
        frame_colors = [get_row_heatmap_colors(row) for _, row in frame_table_df.iterrows()]
        frame_flat_colors = list(map(list, zip(*frame_colors)))  # Transpose

        # Define updated table cells
        updated_cells = dict(
            values=[
                frame_table_df.index,  # Row labels
                *[frame_table_df[col] for col in frame_table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(window_dates)] + frame_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        )

        # Define updated table header
        updated_header = dict(
            values=["Items"] + list(window_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )

        # Calculate the start and end dates for the shaded window
        window_start = window.index.min()
        window_end = window.index.max()

        # Define the shaded rectangle shapes for both plots
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=cpi_df_mom['All Items'].min(),
            y1=cpi_df_mom['All Items'].max(),
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )

        shaded_shape_plot2 = dict(
            type="rect",
            xref="x1",  # Shared x-axis
            yref="y2",  # References the y-axis of the second subplot only
            x0=window_start,
            x1=window_end,
            y0=cpi_df_mom[['Core Goods', 'Core Services', 'Energy', 'Food']].min().min(),
            y1=cpi_df_mom[['Core Goods', 'Core Services', 'Energy', 'Food']].max().max(),
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )

        # Create the frame with updated table and shaded shapes
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1, shaded_shape_plot2]
            ),
            name=f"MoM_{i}",
            traces=[table_trace_index]  # Update the existing table trace
        )
        frames_mom.append(frame)

    # Assign frames to figure
    fig.frames = frames_mom

    # Define slider steps for MoM
    slider_steps_mom = []
    for i in range(len(windows_mom)):
        step = dict(
            method="animate",
            args=[
                [f"MoM_{i}"],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows_mom[i].index.min().strftime("%Y-%m")  # Label represents the start date of the window
        )
        slider_steps_mom.append(step)

    # Optionally, reverse the slider steps to have newest on the left
    slider_steps_mom = slider_steps_mom[::-1]

    # Add sliders to the layout
    sliders_mom = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps_mom,
        visible=True
    )]

    # Update layout with the sliders
    fig.update_layout(
        sliders=sliders_mom
    )

    # Add source annotation
    source_links = (
        "<a href='https://fred.stlouisfed.org/series/CPIAUCSL' style='color:white'>All Items</a>, "
        "<a href='https://fred.stlouisfed.org/series/CPILFESL' style='color:white'>Core CPI</a>, "
        "<a href='https://fred.stlouisfed.org/series/CPIUFDNS' style='color:white'>Food</a>, "
        "<a href='https://fred.stlouisfed.org/series/CPIENGSL' style='color:white'>Energy</a>, "
        "<a href='https://fred.stlouisfed.org/series/CUSR0000SASLE' style='color:white'>Core Services</a>, "
        "<a href='https://fred.stlouisfed.org/series/CUSR0000SACL1E' style='color:white'>Core Goods</a>"
    )

    fig.add_annotation(
        text=("Source: Federal Reserve Economic Data (FRED). " + source_links),
        font=dict(size=8, color="white"),
        xref="paper", yref="paper",
        x=0.5, y=-0.03,  # Position the source annotation at the bottom
        showarrow=False
    )

    # Update layout with dynamic title and other aesthetics
    fig.update_layout(
        title="CPI Month-over-Month (MoM) Changes",
        title_font=dict(size=12),
        width=1000,
        height=900,  # Increased height for larger plots
        margin=dict(t=80, b=80, l=20, r=20),  # Adjusted side margins
        template='plotly_dark',
        barmode='relative',  # Enable stacked bars for the component plot
        showlegend=True,
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="top",    # Align to top
            y=1.05,           # Set the legend position slightly above the plots
            xanchor="center", # Center the legend
            x=0.5,
            font=dict(size=8)  # Set font size for the legend
        ),
        font=dict(size=8)  # Global font size for axis labels, tick marks, etc.
    )

    # Set initial y-axis ranges for MoM
    fig.update_yaxes(range=[cpi_df_mom[['All Items', 'Core CPI']].min().min(), cpi_df_mom[['All Items', 'Core CPI']].max().max()], row=1, col=1)
    fig.update_yaxes(range=[cpi_df_mom[['Core Goods', 'Core Services', 'Energy', 'Food']].min().min(), cpi_df_mom[['Core Goods', 'Core Services', 'Energy', 'Food']].max().max()], row=2, col=1)

    return fig


# In[4]:


def generate_cpi_detailed_graph_3():
    # Initialize Fred API with your API key
    fred = Fred(api_key=api_key)  # Replace with your actual FRED API key

    # Define the FRED series IDs for the different CPI components
    series_ids = {
        'All Items': 'CPIAUCSL',         # All Items CPI
        'Core CPI': 'CPILFESL',          # Core CPI (less food and energy)
        'Food': 'CPIUFDNS',              # Food CPI
        'Energy': 'CPIENGSL',            # Energy CPI
        'Core Goods': 'CUSR0000SACL1E',  # Core Goods
        'Core Services': 'CUSR0000SASLE' # Core Services
    }

    # Fetch data for each series from FRED and store in a DataFrame
    cpi_data = {name: fred.get_series(series_id) for name, series_id in series_ids.items()}
    df = pd.DataFrame(cpi_data)

    # Ensure the DataFrame index is in datetime format
    df.index = pd.to_datetime(df.index)

    # Define relative importance weights (in percentages)
    relative_importance = {
        "Core Services": 61,   # Included in Core CPI
        "Core Goods": 18.4,    # Included in Core CPI
        "Food": 13.4,
        "Energy": 6.9,
        "All Items": 100.0     # Already includes Food, Energy, and Core CPI
    }

    # Define the display start date
    display_start_date = datetime(2015, 1, 1)

    # Calculate MoM percentage changes
    df_mom = df.pct_change(periods=1) * 100

    # Drop rows with NaN values after calculations
    df_mom = df_mom.dropna()

    # Filter data from display_start_date onwards
    df_mom = df_mom[df_mom.index >= display_start_date]

    # Calculate contributions by multiplying percentage changes by relative importance
    contribution_mom = df_mom.copy()
    for column in ['Energy', 'Food', 'Core Goods', 'Core Services', 'All Items']:
        contribution_mom[column] = contribution_mom[column] * relative_importance[column] / 100

    # Round the contribution data to 2 decimal places for plotting
    contribution_mom = contribution_mom.round(2)

    # Create subplots: main plot and table
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        specs=[[{"type": "scatter"}], [{"type": "table"}]]
    )

    # Plot 1: Lines for All Items and Core CPI (MoM)
    fig.add_trace(go.Scatter(
        x=contribution_mom.index, y=contribution_mom['All Items'],
        mode='lines', name='All Items CPI (MoM)',
        line=dict(color='white')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=contribution_mom.index, y=contribution_mom['Core CPI'],
        mode='lines', name='Core CPI (MoM)',
        line=dict(color='magenta', dash='dot')
    ), row=1, col=1)

    # Plot 1: Stacked Bar Plot for Contributions (MoM)
    fig.add_trace(go.Bar(
        x=contribution_mom.index, y=contribution_mom['Core Services'],
        name='Core Services', marker_color='green'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_mom.index, y=contribution_mom['Core Goods'],
        name='Core Goods', marker_color='purple'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_mom.index, y=contribution_mom['Food'],
        name='Food', marker_color='blue'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_mom.index, y=contribution_mom['Energy'],
        name='Energy', marker_color='orange'
    ), row=1, col=1)

    # Ensure barmode is 'relative' for stacked bars
    fig.update_layout(barmode='relative')

    # Prepare data for the table (MoM)
    window_size = 30  # Number of months in each window for the table
    data_length_mom = len(contribution_mom)
    max_start_index_mom = data_length_mom - window_size

    # Generate windows for MoM
    windows_mom = [contribution_mom.iloc[i:i + window_size] for i in range(max_start_index_mom + 1)]

    # Function to generate colors for each row based on values
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 - norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Prepare the initial table data (MoM)
    initial_window = windows_mom[0]
    table_df = initial_window[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].round(2)
    table_df = table_df.T  # Transpose for desired format
    table_header_dates = table_df.columns.strftime("%y-%m")  # Format dates as 'YY-MM'

    # Generate colors for the initial table
    colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
    flat_colors = list(map(list, zip(*colors)))  # Transpose to match Plotly's requirement

    # Add the initial table
    table_trace = go.Table(
        header=dict(
            values=["Item", "Relative Importance (%)"] + list(table_header_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_df.index,  # Row labels (Item names)
                [relative_importance.get(item, 0) for item in table_df.index],  # Relative Importance (%)
                *[table_df[col] for col in table_df.columns]  # Contribution values
            ],
            fill_color=[['black']] * 2 + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'] + ['center'] * len(table_header_dates),
            height=20
        ),
        columnwidth=[4, 4] + [1] * len(table_header_dates)  # Adjust column widths
    )
    fig.add_trace(table_trace, row=2, col=1)  # Table is the second subplot

    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added

    # Generate frames for MoM
    frames = []
    for i, window in enumerate(windows_mom):
        frame_table_df = window[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].round(2)
        frame_table_df = frame_table_df.T  # Transpose
        frame_header_dates = frame_table_df.columns.strftime("%y-%m")  # Format dates as 'YY-MM'

        # Generate colors for the table
        frame_colors = [get_row_heatmap_colors(row) for _, row in frame_table_df.iterrows()]
        frame_flat_colors = list(map(list, zip(*frame_colors)))  # Transpose

        # Define updated table cells
        updated_cells = dict(
            values=[
                frame_table_df.index,  # Row labels (Item names)
                [relative_importance.get(item, 0) for item in frame_table_df.index],  # Relative Importance (%)
                *[frame_table_df[col] for col in frame_table_df.columns]  # Contribution values
            ],
            fill_color=[['black']] * 2 + frame_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'] + ['center'] * len(frame_header_dates),
            height=20
        )

        # Define updated table header
        updated_header = dict(
            values=["Item", "Relative Importance (%)"] + list(frame_header_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )

        # Calculate the start and end dates for the shaded window
        window_start = window.index.min()
        window_end = window.index.max()

        # Define the shaded rectangle shape for the plot
        shaded_shape = dict(
            type="rect",
            xref="x",  # Shared x-axis
            yref="y",  # References the y-axis of the first subplot
            x0=window_start,
            x1=window_end,
            y0=contribution_mom[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].min().min(),
            y1=contribution_mom[['All Items', 'Core CPI', 'Core Goods', 'Core Services', 'Food', 'Energy']].max().max(),
            fillcolor="grey",
            opacity=0.2,
            layer="below",
            line_width=0
        )

        # Create the frame
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape]
            ),
            name=str(i),
            traces=[table_trace_index]
        )
        frames.append(frame)

    fig.frames = frames

    # Define slider steps for MoM
    slider_steps_mom = []
    for i in range(len(windows_mom)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows_mom[i].index.min().strftime("%y-%m")
        )
        slider_steps_mom.append(step)

    # Reverse the slider steps to have the newest on the left
    slider_steps_mom = slider_steps_mom[::-1]

    # Create slider
    sliders_mom = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps_mom,
    )]

    # Add slider to the layout
    fig.update_layout(
        sliders=sliders_mom
    )

    # Add footnote with a link to BLS CPI report
    fig.add_annotation(
        text="Source: <a href='https://www.bls.gov/news.release/pdf/cpi.pdf' style='color:white;' target='_blank'>BLS CPI Report</a> (Weights updated manually)",
        showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.05,
        xanchor='center', yanchor='top',
        font=dict(size=8),
        align='center'
    )

    # Update layout
    fig.update_layout(
        barmode='relative',
        height=900,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=8)
        ),
        font=dict(size=8),
        margin=dict(l=20, r=20, t=80, b=80),
        title="CPI Contributions (Month-over-Month Changes)"
    )

    # Set initial y-axis range for MoM
    fig.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
    fig.update_yaxes(title_text="Contributions (percentage points)", row=1, col=1)

    return fig



# In[5]:


def generate_cpi_detailed_graph_5():
    # Initialize FRED with the provided API Key
    fred = Fred(api_key=api_key)
    
    # Retrieve PCEPI and PCEPILFE data
    pcepi_data = fred.get_series('PCEPI')
    pcepilfe_data = fred.get_series('PCEPILFE')
    
    # Resample data monthly and compute the six-month annualized growth rate
    def annualized_growth_rate(series, window=6):
        return (series.pct_change(window) + 1) ** (12/window) - 1
    
    pcepi_growth = annualized_growth_rate(pcepi_data)
    pcepilfe_growth = annualized_growth_rate(pcepilfe_data)
    
    # Compute YoY percentage change for both series
    pcepi_yoy = pcepi_data.pct_change(periods=12) * 100
    pcepilfe_yoy = pcepilfe_data.pct_change(periods=12) * 100
    
    # Filter data from 2015 onwards
    pcepi_growth = pcepi_growth[pcepi_growth.index >= '2015-01-01']
    pcepilfe_growth = pcepilfe_growth[pcepilfe_growth.index >= '2015-01-01']
    pcepi_yoy = pcepi_yoy[pcepi_yoy.index >= '2015-01-01']
    pcepilfe_yoy = pcepilfe_yoy[pcepilfe_yoy.index >= '2015-01-01']
    
    # Prepare date range for plotting
    dates = pcepi_growth.index
    
    # Get the most recent readings for both series
    recent_pcepi_value = pcepi_growth.dropna().iloc[-1] * 100
    recent_pcepilfe_value = pcepilfe_growth.dropna().iloc[-1] * 100
    recent_pcepi_yoy_value = pcepi_yoy.dropna().iloc[-1]
    recent_pcepilfe_yoy_value = pcepilfe_yoy.dropna().iloc[-1]
    
    # Prepare data for the table with new order and labels
    table_data = pd.DataFrame({
        "PCEPI YoY (%)": pcepi_yoy.dropna().tail(30).values,
        "PCEPI 6M (%)": pcepi_growth.dropna().tail(30).values * 100,
        "PCEPILFE YoY (%)": pcepilfe_yoy.dropna().tail(30).values,
        "PCEPILFE 6M (%)": pcepilfe_growth.dropna().tail(30).values * 100
    }).T.round(1)
    
    # Extract the dates as column headers
    dates_for_table = pcepi_growth.dropna().tail(30).index.strftime("%y-%m")
    
    # Heatmap color functions
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'
    
    # Generate colors for the table based on the data rows
    colors = [get_row_heatmap_colors(row) for _, row in table_data.iterrows()]
    
    # Transpose the list of colors to match Plotly's requirement (each sublist represents a column of colors)
    flat_colors = list(map(list, zip(*colors)))
    
    # Calculate long-term averages for PCEPI and PCEPILFE growth rates
    average_pcepi_growth = pcepi_growth.mean() * 100  # Convert to percentage
    average_pcepilfe_growth = pcepilfe_growth.mean() * 100  # Convert to percentage
    
    # Create subplots: 2 columns for charts and 1 row for the table
    fig = make_subplots(
        rows=2, cols=2, 
        row_heights=[0.6, 0.4],  # 60% for charts, 40% for table
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "table", "colspan": 2}, None]],  # Use full width for the table
        subplot_titles=('PCEPI', 'PCEPILFE'),
        vertical_spacing=0.05  # Smaller gap between plots and table
    )
    
    # Add PCEPI chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=pcepi_growth * 100, mode='lines', name='6M Annualized PCEPI', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_pcepi_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=1)
    
    # Add PCEPILFE chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=pcepilfe_growth * 100, mode='lines', name='6M Annualized PCEPILFE', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_pcepilfe_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=2)
    
    # Add YoY PCEPI (White)
    fig.add_trace(go.Scatter(
        x=dates, y=pcepi_yoy, mode='lines', name='YoY PCEPI', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=1)
    
    # Add YoY PCEPILFE (White)
    fig.add_trace(go.Scatter(
        x=dates, y=pcepilfe_yoy, mode='lines', name='YoY PCEPILFE', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=2)
    
    # Add Long-term Average PCEPI line (Orange Dash)
    fig.add_trace(go.Scatter(
        x=dates, y=[average_pcepi_growth]*len(dates), mode='lines', name='Long-term Average PCEPI', 
        line=dict(color='red', dash='dash')
    ), row=1, col=1)
    
    # Add Long-term Average PCEPILFE line (Purple Dash)
    fig.add_trace(go.Scatter(
        x=dates, y=[average_pcepilfe_growth]*len(dates), mode='lines', name='Long-term Average PCEPILFE', 
        line=dict(color='red', dash='dash')
    ), row=1, col=2)
    
    # Add the initial table below the charts
    table = go.Table(
        header=dict(
            values=["Items"] + list(dates_for_table),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_data.index,  # Row labels
                *[table_data[col] for col in table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(dates_for_table)] + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        ),
        columnwidth=[4] + [0.5] * len(dates_for_table)  # Adjusted column widths
    )
    fig.add_trace(table, row=2, col=1)  # Table spans both columns
    
    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added
    
    # Calculate global y-axis range for both plots
    # Combine all y-values from lines and bars to determine global min and max
    y_values = []
    for column in ['pcepi_growth', 'pcepilfe_growth', 'pcepi_yoy', 'pcepilfe_yoy']:
        if column == 'pcepi_growth':
            y_values.extend(pcepi_growth.dropna().values * 100)
        elif column == 'pcepilfe_growth':
            y_values.extend(pcepilfe_growth.dropna().values * 100)
        elif column == 'pcepi_yoy':
            y_values.extend(pcepi_yoy.dropna().values)
        elif column == 'pcepilfe_yoy':
            y_values.extend(pcepilfe_yoy.dropna().values)
    global_y_min = min(y_values)
    global_y_max = max(y_values)
    # Add some padding
    padding = (global_y_max - global_y_min) * 0.05
    global_y_min -= padding
    global_y_max += padding
    
    # Define rolling windows (e.g., 30-month rolling windows)
    window_size = 30  # Number of months in each window
    windows = []
    for i in range(len(pcepi_growth) - window_size + 1):
        window = {
            'pcepi_yoy': pcepi_yoy.iloc[i:i+window_size],
            'pcepi_growth': pcepi_growth.iloc[i:i+window_size],
            'pcepilfe_yoy': pcepilfe_yoy.iloc[i:i+window_size],
            'pcepilfe_growth': pcepilfe_growth.iloc[i:i+window_size],
            'dates': pcepi_growth.index[i:i+window_size]
        }
        windows.append(window)
    
    # Function to prepare table data for a given window
    def prepare_table_data(window):
        window_data = {
            "PCEPI YoY (%)": window['pcepi_yoy'].values,
            "PCEPI 6M (%)": window['pcepi_growth'].values * 100,
            "PCEPILFE YoY (%)": window['pcepilfe_yoy'].values,
            "PCEPILFE 6M (%)": window['pcepilfe_growth'].values * 100
        }
        table_df = pd.DataFrame(window_data).T.round(1)
        window_dates = window['dates'].strftime("%y-%m")
        return table_df, window_dates
    
    # Generate frames for each window
    frames = []
    for i, window in enumerate(windows):
        table_df, window_dates = prepare_table_data(window)
        
        # Generate colors for the table
        window_colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
        window_flat_colors = list(map(list, zip(*window_colors)))  # Transpose
        
        # Define updated table cells
        updated_cells = dict(
            values=[
                table_df.index,  # Row labels
                *[table_df[col] for col in table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(window_dates)] + window_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        )
        
        # Define updated table header
        updated_header = dict(
            values=["Items"] + list(window_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )
        
        # Calculate the start and end dates for the shaded window
        window_start = window['dates'].min()
        window_end = window['dates'].max()
        
        # Define the shaded rectangle shapes for both plots
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        shaded_shape_plot2 = dict(
            type="rect",
            xref="x2",  # References the x-axis of the second subplot
            yref="y2",  # References the y-axis of the second subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        # Create the frame with updated table and shaded shapes
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1, shaded_shape_plot2]  # Add shaded areas to both plots
            ),
            name=str(i),
            traces=[table_trace_index]  # Update the existing table trace
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Define slider steps
    slider_steps = []
    for i in range(len(windows)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows[i]['dates'].min().strftime("%Y-%m")  # Label represents the start date of the window
        )
        slider_steps.append(step)
    
    # Optionally, reverse the slider steps to have newest on the left
    slider_steps = slider_steps[::-1]
    
    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]
    
    # Add the slider to the layout
    fig.update_layout(
        sliders=sliders
    )
    
    # Add annotations for the latest values on both plots
    fig.add_annotation(
        x=dates[-1], y=recent_pcepi_value, xref="x1", yref="y1", 
        text=f"Latest: {recent_pcepi_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    fig.add_annotation(
        x=dates[-1], y=recent_pcepilfe_value, xref="x2", yref="y2", 
        text=f"Latest: {recent_pcepilfe_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    # Add source annotation at the bottom of the chart with hyperlinks
    fig.add_annotation(
        text=('Source: Federal Reserve Economic Data (FRED). '
              '<a href="https://fred.stlouisfed.org/series/PCEPI" style="color: white">PCEPI</a>, '
              '<a href="https://fred.stlouisfed.org/series/PCEPILFE" style="color: white">PCEPILFE</a>.'),
        font=dict(size=8, color="white"),  # Font size and color
        xref="paper", yref="paper",  # Use paper coordinates
        x=0.5, y=-0.03,  # Position the source annotation at the bottom
        showarrow=False
    )
    
    # Update layout and labels
    fig.update_layout(
        title="",
        yaxis_title="Growth Rate (%)",
        yaxis2_title="Growth Rate (%)",
        template="plotly_dark",
        font=dict(color="white", size=8),
        title_x=0.5,
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="top",  # Anchor the legend to the top
            y=1.2,  # Set the legend position slightly below the title
            xanchor="center",  # Center the legend
            x=0.5,
            font=dict(size=8)  # Set the font size for the legend
        ),
        height=600,  # Adjusted height
        width=1000,  # Adjusted width
        margin=dict(t=50, b=50, l=20, r=20)  # Adjusted side margins
    )
    
    # Fix the y-axis range for both plots to prevent dynamic scaling
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=1)
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=2)
    
    # Update Y-axis labels to reflect that we're displaying Growth Rates (percentage points)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
    
    return fig


# In[6]:


def generate_cpi_detailed_graph_4(start_date='2020-01-01'):
    # Initialize FRED with the provided API Key
    fred = Fred(api_key=api_key)

    # Define the series IDs (expanded with Rent added)
    series_ids = {
        'All Items CPI': 'CPIAUCSL',        # All Items CPI
        'Core CPI': 'CPILFESL',             # Core CPI (less food and energy)
        'Food': 'CPIUFDNS',                 # Food CPI
        'Energy': 'CPIENGSL',               # Energy CPI
        'Core Goods': 'CUSR0000SACL1E',     # Core Goods CPI
        'Core Services': 'CUSR0000SASLE',   # Core Services CPI
        'Rent': 'CUSR0000SEHA'              # Rent CPI
    }

    # Fetch data from FRED for the past two years with error handling
    try:
        data = {name: fred.get_series(series_id, start=start_date) for name, series_id in series_ids.items()}
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate the 6-month annualized growth rates
    growth_rates_6m = df.pct_change(periods=6).apply(lambda x: ((1 + x) ** 2) - 1) * 100

    # Align growth rates with months
    growth_rates_6m = growth_rates_6m.dropna()

    # Get the latest date
    latest_date = growth_rates_6m.index[-1].strftime('%Y-%m-%d')

    # Prepare data for plotting, including 6M Ago, 12M Ago, 18M Ago, and the Latest Data Point
    plot_data = {
        '18M Ago': growth_rates_6m.iloc[-18] if len(growth_rates_6m) > 18 else None,
        '12M Ago': growth_rates_6m.iloc[-12] if len(growth_rates_6m) > 12 else None,
        '6M Ago': growth_rates_6m.iloc[-6] if len(growth_rates_6m) > 6 else None,
        f'Latest Data Point ({latest_date})': growth_rates_6m.iloc[-1]
    }

    # Filter out None values from plot_data
    plot_data = {k: v for k, v in plot_data.items() if v is not None}

    # Create DataFrame for easier plotting
    plot_df = pd.DataFrame(plot_data)

    # Plotting with Plotly
    fig = go.Figure()

    # Add bars for each time period with numbers on top of each bar
    colors = ['#A67D5D', '#708090', '#B0C4DE', '#4682B4', '#2F4F4F']
    for i, period in enumerate(plot_df.columns):
        fig.add_trace(go.Bar(
            x=plot_df.index,
            y=plot_df[period],
            name=period,
            marker_color=colors[i % len(colors)],
            width=0.2,  # Adjust width based on the number of bars
            text=plot_df[period].round(1),  # Show the values on top of the bars with 1 decimal place
            textposition='outside',         # Display the text outside the bar
            texttemplate='%{text:.1f}',     # Format the text to 1 decimal place
            textfont_size=8                # Set font size to 12
        ))

    # Update layout to match the style with more neutral colors and space
    fig.update_layout(
        yaxis_title='Inflation Rate (%)',
        xaxis_title='',
        barmode='group',
        template='plotly_dark',
        font=dict(color='white'),
        width=1000,
        height=500,
        title_font=dict(size=8),
        bargap=0.2,  # Adjust gap between bar groups for more spacing
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )

    return fig


# In[7]:


def generate_cpi_detailed_graph_6():
    # Initialize FRED API
    fred = Fred(api_key=api_key)

    # Fetch Core PCE data starting from December 2022 to get accurate MoM for January 2023
    data = fred.get_series('PCEPILFE', observation_start='2022-12-01')

    # Calculate month-over-month changes (annualized)
    core_pce_mom_ann = data.pct_change().dropna() * 12 * 100

    # Filter the data for 2023 and 2024
    data_2023_2024 = core_pce_mom_ann['2023-01-01':]

    # Define months
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Extract data for 2023
    core_pce_2023 = data_2023_2024['2023'].reindex(
        pd.PeriodIndex(['2023-{:02d}-01'.format(m) for m in range(1,13)], freq='M').to_timestamp()
    ).tolist()

    # Extract data for 2024
    core_pce_2024 = data_2023_2024['2024'].reindex(
        pd.PeriodIndex(['2024-{:02d}-01'.format(m) for m in range(1,13)], freq='M').to_timestamp()
    ).tolist()

    # Handle missing data by replacing NaN with None
    core_pce_2023 = [x if pd.notnull(x) else None for x in core_pce_2023]
    core_pce_2024 = [x if pd.notnull(x) else None for x in core_pce_2024]

    # Calculate differences where data for both years is available
    core_pce_diff = [
        round(core_pce_2024[i] - core_pce_2023[i], 2) if (core_pce_2023[i] is not None and core_pce_2024[i] is not None) else None
        for i in range(12)
    ]

    # Calculate averages
    avg_2023 = round(sum(filter(None, core_pce_2023)) / len(list(filter(None, core_pce_2023))), 2)
    avg_2024 = round(sum(filter(None, core_pce_2024)) / len(list(filter(None, core_pce_2024))), 2)
    avg_diff = round(avg_2024 - avg_2023, 2)

    # Prepare display data with two decimal places
    core_pce_2023_display = [f"{x:.2f}" if x is not None else "" for x in core_pce_2023]
    core_pce_2024_display = [f"{x:.2f}" if x is not None else "" for x in core_pce_2024]
    core_pce_diff_display = [f"{x:.2f}" if x is not None else "" for x in core_pce_diff]

    # Append averages to the display lists
    months_to_display = months + ['Avg to Last Month']
    core_pce_2023_display.append(f"{avg_2023:.2f}")
    core_pce_2024_display.append(f"{avg_2024:.2f}")
    core_pce_diff_display.append(f"{avg_diff:.2f}")

    # Append averages to data lists (they won't be plotted)
    core_pce_2023.append(avg_2023)
    core_pce_2024.append(avg_2024)

    # Create subplots: one for the bar chart and one for the table
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.4],  # Adjust heights as needed
        specs=[[{"type": "bar"}], [{"type": "table"}]],
        subplot_titles=["Core PCE (M/M, Annualized) Comparison: 2023 vs 2024", ""]
    )

    # Add Bar for 2023
    fig.add_trace(
        go.Bar(
            x=months,
            y=core_pce_2023[:-1],  # Exclude average
            name='2023',
            marker_color='rgba(0, 170, 255, 0.7)'
        ),
        row=1, col=1
    )

    # Add Bar for 2024
    fig.add_trace(
        go.Bar(
            x=months,
            y=core_pce_2024[:-1],  # Exclude average
            name='2024',
            marker_color='rgba(255, 165, 0, 0.7)'
        ),
        row=1, col=1
    )

    # Update layout for grouped bars
    fig.update_layout(
        barmode='group',
        title_text="Core PCE, Month-over-Month Annualized Changes",
        template="plotly_dark",
        font=dict(color='white'),
        height=800,  # Adjust height as needed
        margin=dict(l=50, r=50, t=100, b=100),
        legend=dict(x=0.8, y=1.15, orientation='h')
    )

    # Prepare font sizes for the table
    # Assuming 13 rows (12 months + average) and 4 columns
    num_rows = len(months_to_display)
    num_cols = 4
    main_font_size = 10
    diff_font_size = 8

    # Create a list of font sizes: [main, main, main, diff] for each row
    font_size_list = [main_font_size, main_font_size, main_font_size, diff_font_size] * num_rows

    # Create the table
    table = go.Table(
        header=dict(
            values=['Month', '2023', '2024', 'Diff'],
            fill_color='#1f1f1f',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[months_to_display, core_pce_2023_display, core_pce_2024_display, core_pce_diff_display],
            fill_color='black',
            align='center',
            font=dict(color='white', size=10),  # Default font size
            # Apply the font sizes list to the 'Diff' column
            # Since Plotly doesn't support per-column font sizes directly,
            # we simulate it by creating a flat list where 'Diff' cells have smaller font
            font_size=font_size_list
        )
    )

    # Add table to the second row
    fig.add_trace(table, row=2, col=1)

    # Add footnote with a link to PCE data and source text
    fig.add_annotation(
        text=("Source: <a href='https://fred.stlouisfed.org/series/PCEPILFE' style='color:white;' target='_blank'>FRED</a>"),
        showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.25,
        xanchor='center', yanchor='top',
        font=dict(size=10),
        align='center'
    )

    return fig


# In[8]:


def generate_cpi_detailed_graph_7():
    # Initialize FRED with the provided API Key
    fred = Fred(api_key=api_key)
    
    # Retrieve PPI data
    ppiaco_data = fred.get_series('PPIACO')  # Producer Price Index for All Commodities
    ppifes_data = fred.get_series('PPIFES')  # Producer Price Index for Finished Goods
    
    # Resample data monthly and compute the six-month annualized growth rate
    def annualized_growth_rate(series, window=6):
        return (series.pct_change(window) + 1) ** (12/window) - 1
    
    ppiaco_growth = annualized_growth_rate(ppiaco_data)
    ppifes_growth = annualized_growth_rate(ppifes_data)
    
    # Compute YoY percentage change for both series
    ppiaco_yoy = ppiaco_data.pct_change(periods=12) * 100
    ppifes_yoy = ppifes_data.pct_change(periods=12) * 100
    
    # Filter data from 2015 onwards
    ppiaco_growth = ppiaco_growth[ppiaco_growth.index >= '2015-01-01']
    ppifes_growth = ppifes_growth[ppifes_growth.index >= '2015-01-01']
    ppiaco_yoy = ppiaco_yoy[ppiaco_yoy.index >= '2015-01-01']
    ppifes_yoy = ppifes_yoy[ppifes_yoy.index >= '2015-01-01']
    
    # Prepare date range for plotting
    dates = ppiaco_growth.index
    
    # Get the most recent readings for both series
    recent_ppiaco_value = ppiaco_growth.dropna().iloc[-1] * 100
    recent_ppifes_value = ppifes_growth.dropna().iloc[-1] * 100
    recent_ppiaco_yoy_value = ppiaco_yoy.dropna().iloc[-1]
    recent_ppifes_yoy_value = ppifes_yoy.dropna().iloc[-1]
    
    # Compute long-term averages
    ppiaco_long_term_avg = ppiaco_growth.mean() * 100
    ppifes_long_term_avg = ppifes_growth.mean() * 100
    
    # Prepare data for the table with new order and labels
    table_data = pd.DataFrame({
        "PPI All Comm YoY (%)": ppiaco_yoy.dropna().tail(30).values,
        "PPI All Comm 6M (%)": ppiaco_growth.dropna().tail(30).values * 100,
        "PPI Final Demand YoY (%)": ppifes_yoy.dropna().tail(30).values,
        "PPI Final Demand 6M (%)": ppifes_growth.dropna().tail(30).values * 100
    }).T.round(1)
    
    # Extract the dates as column headers
    dates_for_table = ppiaco_growth.dropna().tail(30).index.strftime("%y-%m")
    
    # Heatmap color functions
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'
    
    # Generate colors for the table based on the data rows
    colors = [get_row_heatmap_colors(row) for _, row in table_data.iterrows()]
    
    # Transpose the list of colors to match Plotly's requirement (each sublist represents a column of colors)
    flat_colors = list(map(list, zip(*colors)))
    
    # Create subplots: 2 columns for charts and 1 row for the table
    fig = make_subplots(
        rows=2, cols=2, 
        row_heights=[0.6, 0.4],  # 60% for charts, 40% for table
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "table", "colspan": 2}, None]],  # Use full width for the table
        subplot_titles=('PPI All Commodities', 'PPI Final Demand'),
        vertical_spacing=0.05  # Smaller gap between plots and table
    )
    
    # Add PPI All Commodities chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=ppiaco_growth * 100, mode='lines', name='6M Annualized PPI All Comm', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_ppiaco_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=1)
    
    # Add PPI Final Demand chart (6M Annualized, Light Blue)
    fig.add_trace(go.Scatter(
        x=dates, y=ppifes_growth * 100, mode='lines', name='6M Annualized PPI Final Demand', 
        line=dict(color='cyan', width=2),
        text=[f"Latest: {recent_ppifes_value:.2f}%" for _ in dates],
        hoverinfo='text+y'
    ), row=1, col=2)
    
    # Add YoY PPI All Commodities (White)
    fig.add_trace(go.Scatter(
        x=dates, y=ppiaco_yoy, mode='lines', name='YoY PPI All Comm', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=1)
    
    # Add YoY PPI Final Demand (White)
    fig.add_trace(go.Scatter(
        x=dates, y=ppifes_yoy, mode='lines', name='YoY PPI Final Demand', 
        line=dict(color='white', width=1),
        hoverinfo='none'
    ), row=1, col=2)
    
    # Add the Long Term Average line to both charts
    fig.add_trace(go.Scatter(
        x=dates, y=[ppiaco_long_term_avg]*len(dates), mode='lines', name='Long Term Average', 
        line=dict(color='crimson', dash='dash')
    ), row=1, col=1)
    
    # Duplicate the Long Term Average line for the second chart without showing it in the legend
    fig.add_trace(go.Scatter(
        x=dates, y=[ppifes_long_term_avg]*len(dates), mode='lines', name='', showlegend=False, 
        line=dict(color='crimson', dash='dash')
    ), row=1, col=2)
    
    # Add the initial table below the charts
    table = go.Table(
        header=dict(
            values=["Items"] + list(dates_for_table),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_data.index,  # Row labels
                *[table_data[col] for col in table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(dates_for_table)] + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        ),
        columnwidth=[4] + [0.5] * len(dates_for_table)  # Adjusted column widths
    )
    fig.add_trace(table, row=2, col=1)  # Table spans both columns
    
    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added
    
    # Calculate global y-axis range for both plots
    # Combine all y-values from lines and bars to determine global min and max
    y_values = []
    for column in ['ppiaco_growth', 'ppifes_growth', 'ppiaco_yoy', 'ppifes_yoy']:
        if column == 'ppiaco_growth':
            y_values.extend(ppiaco_growth.dropna().values * 100)
        elif column == 'ppifes_growth':
            y_values.extend(ppifes_growth.dropna().values * 100)
        elif column == 'ppiaco_yoy':
            y_values.extend(ppiaco_yoy.dropna().values)
        elif column == 'ppifes_yoy':
            y_values.extend(ppifes_yoy.dropna().values)
    global_y_min = min(y_values)
    global_y_max = max(y_values)
    # Add some padding
    padding = (global_y_max - global_y_min) * 0.05
    global_y_min -= padding
    global_y_max += padding
    
    # Define rolling windows (e.g., 30-month rolling windows)
    window_size = 30  # Number of months in each window
    windows = []
    for i in range(len(ppiaco_growth) - window_size + 1):
        window = {
            'ppiaco_yoy': ppiaco_yoy.iloc[i:i+window_size],
            'ppiaco_growth': ppiaco_growth.iloc[i:i+window_size],
            'ppifes_yoy': ppifes_yoy.iloc[i:i+window_size],
            'ppifes_growth': ppifes_growth.iloc[i:i+window_size],
            'dates': ppiaco_growth.index[i:i+window_size]
        }
        windows.append(window)
    
    # Function to prepare table data for a given window
    def prepare_table_data(window):
        window_data = {
            "PPI All Comm YoY (%)": window['ppiaco_yoy'].values,
            "PPI All Comm 6M (%)": window['ppiaco_growth'].values * 100,
            "PPI Final Demand YoY (%)": window['ppifes_yoy'].values,
            "PPI Final Demand 6M (%)": window['ppifes_growth'].values * 100
        }
        table_df = pd.DataFrame(window_data).T.round(1)
        window_dates = window['dates'].strftime("%y-%m")
        return table_df, window_dates
    
    # Generate frames for each window
    frames = []
    for i, window in enumerate(windows):
        table_df, window_dates = prepare_table_data(window)
        
        # Generate colors for the table
        window_colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
        window_flat_colors = list(map(list, zip(*window_colors)))  # Transpose
        
        # Define updated table cells
        updated_cells = dict(
            values=[
                table_df.index,  # Row labels
                *[table_df[col] for col in table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(window_dates)] + window_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        )
        
        # Define updated table header
        updated_header = dict(
            values=["Items"] + list(window_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )
        
        # Calculate the start and end dates for the shaded window
        window_start = window['dates'].min()
        window_end = window['dates'].max()
        
        # Define the shaded rectangle shapes for both plots
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        shaded_shape_plot2 = dict(
            type="rect",
            xref="x2",  # References the x-axis of the second subplot
            yref="y2",  # References the y-axis of the second subplot only
            x0=window_start,
            x1=window_end,
            y0=global_y_min,
            y1=global_y_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        # Create the frame with updated table and shaded shapes
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1, shaded_shape_plot2]  # Add shaded areas to both plots
            ),
            name=str(i),
            traces=[table_trace_index]  # Update the existing table trace
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Define slider steps
    slider_steps = []
    for i in range(len(windows)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows[i]['dates'].min().strftime("%Y-%m")  # Label represents the start date of the window
        )
        slider_steps.append(step)
    
    # Optionally, reverse the slider steps to have newest on the left
    slider_steps = slider_steps[::-1]
    
    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]
    
    # Add the slider to the layout
    fig.update_layout(
        sliders=sliders
    )
    
    # Add annotations for the latest values on both plots
    fig.add_annotation(
        x=dates[-1], y=recent_ppiaco_value, xref="x1", yref="y1", 
        text=f"Latest: {recent_ppiaco_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    fig.add_annotation(
        x=dates[-1], y=recent_ppifes_value, xref="x2", yref="y2", 
        text=f"Latest: {recent_ppifes_value:.2f}%", showarrow=False,
        font=dict(size=8, color="cyan"),
        xanchor="center", yanchor="bottom",  # Align tag above the line
        ax=0, ay=40  # Shift the tag 40 pixels up
    )
    
    # Add source annotation at the bottom of the chart with hyperlinks
    fig.add_annotation(
        text=('Source: Federal Reserve Economic Data (FRED). '
              '<a href="https://fred.stlouisfed.org/graph/?g=1tHOq" style="color: white">PPI: Final Demand</a>, '
              '<a href="https://fred.stlouisfed.org/graph/?g=1tHOy" style="color: white">PPI: All Comm</a>.'),
        font=dict(size=8, color="white"),  # Font size and color
        xref="paper", yref="paper",  # Use paper coordinates
        x=0.5, y=-0.03,  # Position the source annotation at the bottom
        showarrow=False
    )
    
    # Update layout and labels
    fig.update_layout(
        title="",
        yaxis_title="Growth Rate (%)",
        yaxis2_title="Growth Rate (%)",
        template="plotly_dark",
        font=dict(color="white", size=8),
        title_x=0.5,
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="top",  # Anchor the legend to the top
            y=1.2,  # Set the legend position slightly below the title
            xanchor="center",  # Center the legend
            x=0.5,
            font=dict(size=8)  # Set the font size for the legend
        ),
        height=600,  # Adjusted height
        width=1000,  # Adjusted width
        margin=dict(t=50, b=50, l=20, r=20)  # Adjusted side margins
    )
    
    # Fix the y-axis range for both plots to prevent dynamic scaling
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=1)
    fig.update_yaxes(range=[global_y_min, global_y_max], row=1, col=2)
    
    # Update Y-axis labels to reflect that we're displaying Contributions (percentage points)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
    
    return fig


# In[9]:


def generate_cpi_detailed_graph_8(selected_period='MoM', months_for_table=30):
    # Initialize Fred API with your API key
    fred = Fred(api_key='7227512392e5e5d2a2679a261d2bb3a9')  # Ensure your API key is secure
    
    # Define the FRED series IDs for the different PPI components
    series_ids = {
        'All Items PPI': 'PPIFID',            # All Items PPI
        'Core PPI': 'WPUFD413',               # Core PPI (Final Demand Goods Less Foods and Energy)
        'Foods PPI': 'PPIFDF',                # Foods PPI
        'Energy PPI': 'PPIFDE',               # Energy PPI
        'Goods PPI': 'PPIFDG',                # Goods PPI (Final Demand)
        'Services PPI': 'PPIFDS',             # Services PPI
        'Construction PPI': 'PPIFDC'          # Construction PPI
    }
    
    # Fetch the data from FRED
    ppi_data = {name: fred.get_series(series_id) for name, series_id in series_ids.items()}
    
    # Convert the data into a DataFrame
    ppi_df = pd.DataFrame(ppi_data)
    
    # Ensure the DataFrame index is in datetime format
    ppi_df.index = pd.to_datetime(ppi_df.index)
    
    # Filter data from February 2015 onwards
    ppi_df = ppi_df[ppi_df.index >= '2015-02-01']
    
    # Determine y-axis ranges and other parameters based on selected_period
    if selected_period == 'MoM':
        required_prior_months = 1  # To compute MoM for the first display month
        window_size = months_for_table  # Number of months in each window
        y1_min, y1_max = -5, 5         # Y-axis range for the first plot
        y2_min, y2_max = -20, 20       # Y-axis range for the second plot
        title = "Current graph displays Month-over-Month (MoM) Changes"
    elif selected_period == 'YoY':
        required_prior_months = 12  # To compute YoY for the first display month
        window_size = months_for_table  # Number of months in each window
        y1_min, y1_max = -15, 15      # Y-axis range for the first plot
        y2_min, y2_max = -30, 30      # Y-axis range for the second plot
        title = "Current graph displays Year-over-Year (YoY) Changes"
    else:
        raise ValueError("selected_period must be either 'MoM' or 'YoY'.")
    
    # Calculate the start date needed to accommodate the buffer
    display_start_date = ppi_df.index.min()
    buffer_start_date = display_start_date - relativedelta(months=required_prior_months)
    
    # Fetch data starting from buffer_start_date to ensure calculations are possible
    ppi_df = ppi_df[ppi_df.index >= buffer_start_date]
    
    # Calculate the Month-over-Month or Year-over-Year percentage change based on the selected_period input
    if selected_period == 'MoM':
        data_df = ppi_df.pct_change(periods=1) * 100  # MoM percentage change
        
        # Calculate 3-month moving average (MoM)
        data_df['All Items PPI (3-mo MA)'] = data_df['All Items PPI'].rolling(window=3).mean()
        data_df['Core PPI (3-mo MA)'] = data_df['Core PPI'].rolling(window=3).mean()
        
        # MoM source links
        source_links = (
            "<a href='https://fred.stlouisfed.org/graph/?g=1tRZ4' style='color:white'>All Items PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tRZu' style='color:white'>Core PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tRZH' style='color:white'>Foods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tRZO' style='color:white'>Energy PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS05' style='color:white'>Goods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS0o' style='color:white'>Services PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS0C' style='color:white'>Construction PPI</a>"
        )

    else:  # 'YoY'
        data_df = ppi_df.pct_change(periods=12) * 100  # YoY percentage change
        
        # Use MoM data to calculate the 6-month annualized growth rate
        data_df_mom = ppi_df.pct_change(periods=1)
        
        # Calculate the 6-month annualized rate using MoM data
        data_df['All Items PPI (6-mo Annualized)'] = (
            data_df_mom['All Items PPI'].rolling(window=6).apply(lambda x: np.prod(1 + x) ** (12 / 6) - 1)) * 100
        data_df['Core PPI (6-mo Annualized)'] = (
            data_df_mom['Core PPI'].rolling(window=6).apply(lambda x: np.prod(1 + x) ** (12 / 6) - 1)) * 100

        # YoY source links
        source_links = (
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1p' style='color:white'>All Items PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1r' style='color:white'>Core PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1t' style='color:white'>Foods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1x' style='color:white'>Energy PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1z' style='color:white'>Goods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1D' style='color:white'>Services PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tS1K' style='color:white'>Construction PPI</a>"
        )
    
    # Drop rows with NaN values after calculations
    data_df = data_df.dropna()
    
    # Further filter data to display only from display_start_date onwards
    data_df = data_df[data_df.index >= display_start_date]
    
    # Create subplots: 3 rows (2 for plots, 1 for table)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.35, 0.3],  # Adjust the heights of the rows to allocate more space for the table
        subplot_titles=("Core PPI & All Items PPI %", "Selected Components % (Stacked)", "PPI Table"),
        vertical_spacing=0.05,  # Smaller gap between plots and the table
        specs=[[{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]]
    )
    
    # Plot 1: Core PPI and All Items PPI (Main plot)
    fig.add_trace(go.Scatter(
        x=data_df.index, y=data_df['All Items PPI'],
        mode='lines', name='All Items PPI %',
        line=dict(color='white')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data_df.index, y=data_df['Core PPI'],
        mode='lines', name='Core PPI %',
        line=dict(color='magenta')
    ), row=1, col=1)
    
    # Add 6-month annualized rate (for YoY plot)
    if selected_period == 'YoY':
        fig.add_trace(go.Scatter(
            x=data_df.index, y=data_df['All Items PPI (6-mo Annualized)'],
            mode='lines', name='All Items PPI (6-mo Annualized)',
            line=dict(color='white', dash='dot')
        ), row=1, col=1)
    
        fig.add_trace(go.Scatter(
            x=data_df.index, y=data_df['Core PPI (6-mo Annualized)'],
            mode='lines', name='Core PPI (6-mo Annualized)',
            line=dict(color='magenta', dash='dot')
        ), row=1, col=1)
    
    # Add 3-month moving average (for MoM plot)
    if selected_period == 'MoM':
        fig.add_trace(go.Scatter(
            x=data_df.index, y=data_df['All Items PPI (3-mo MA)'],
            mode='lines', name='All Items PPI (3-mo MA)',
            line=dict(color='white', dash='dot')
        ), row=1, col=1)
    
        fig.add_trace(go.Scatter(
            x=data_df.index, y=data_df['Core PPI (3-mo MA)'],
            mode='lines', name='Core PPI (3-mo MA)',
            line=dict(color='magenta', dash='dot')
        ), row=1, col=1)
    
    # Subplot 2: Stacked bar plot for components (Foods PPI, Energy PPI, Goods PPI, Services PPI, Construction PPI)
    fig.add_trace(go.Bar(
        x=data_df.index, y=data_df['Goods PPI'], name='Goods PPI %', marker_color='purple'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=data_df.index, y=data_df['Services PPI'], name='Services PPI %', marker_color='green'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=data_df.index, y=data_df['Energy PPI'], name='Energy PPI %', marker_color='orange'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=data_df.index, y=data_df['Foods PPI'], name='Foods PPI %', marker_color='blue'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=data_df.index, y=data_df['Construction PPI'], name='Construction PPI %', marker_color='yellow'
    ), row=2, col=1)
    
    # Create a table summarizing % changes for the selected number of months, with the specified order
    table_data = data_df[['All Items PPI', 'Core PPI', 'Goods PPI', 'Services PPI', 'Foods PPI', 'Energy PPI', 'Construction PPI']].tail(months_for_table).round(1)
    
    # Transpose the table for desired format: Dates as column headers, items on the left-hand side
    table_data = table_data.T  # Transpose the dataframe to switch rows and columns
    
    # Function to generate colors for each row based on values
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]
    
    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'
    
    # Generate colors for the table
    colors = [get_row_heatmap_colors(row) for _, row in table_data.iterrows()]
    
    # Transpose the list of colors to match Plotly's requirement (each sublist represents a column of colors)
    flat_colors = list(map(list, zip(*colors)))
    
    # Add the initial table
    table_trace = go.Table(
        header=dict(
            values=["Items"] + list(table_data.columns.strftime("%y-%m")),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_data.index,  # Row labels
                *[table_data[col] for col in table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(table_data.columns)] + flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        ),
        columnwidth=[2] + [1] * len(table_data.columns)  # Ensure uniform column width
    )
    fig.add_trace(table_trace, row=3, col=1)  # Table is the third subplot
    
    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added
    
    # Function to prepare table data for a given window
    def prepare_table_data(window):
        window_data = window[['All Items PPI', 'Core PPI', 'Goods PPI', 'Services PPI', 'Foods PPI', 'Energy PPI', 'Construction PPI']].round(1)
        window_df = window_data.copy()
        window_df = window_df.T  # Transpose for desired format
        window_dates = window_df.columns.strftime("%y-%m")
        return window_df, window_dates
    
    # Generate all possible windows starting from display_start_date
    windows = [data_df.iloc[i:i + window_size] for i in range(len(data_df) - window_size + 1)]
    
    # Function to generate colors for each row based on values (for frames)
    def get_row_heatmap_colors_frame(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]
    
    # Generate frames for the slider
    frames = []
    for i, window in enumerate(windows):
        frame_table_df, window_dates = prepare_table_data(window)
        
        # Generate colors for the table
        frame_colors = [get_row_heatmap_colors_frame(row) for _, row in frame_table_df.iterrows()]
        frame_flat_colors = list(map(list, zip(*frame_colors)))  # Transpose to match Plotly's requirement
        
        # Define updated table cells
        updated_cells = dict(
            values=[
                frame_table_df.index,  # Row labels
                *[frame_table_df[col] for col in frame_table_df.columns]  # Column data
            ],
            fill_color=[['black'] * len(window_dates)] + frame_flat_colors,
            font=dict(color='white', size=8),
            align=['left', 'center'],
            height=20
        )
        
        # Define updated table header
        updated_header = dict(
            values=["Items"] + list(window_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )
        
        # Calculate the start and end dates for the shaded window
        window_start = window.index.min()
        window_end = window.index.max()
        
        # Define the shaded rectangle shapes for both plots with fixed y-axis ranges
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=y1_min,
            y1=y1_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        shaded_shape_plot2 = dict(
            type="rect",
            xref="x1",  # Shared x-axis
            yref="y2",  # References the y-axis of the second subplot only
            x0=window_start,
            x1=window_end,
            y0=y2_min,
            y1=y2_max,
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )
        
        # Create the frame with updated table and shaded shapes
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1, shaded_shape_plot2]
            ),
            name=str(i),
            traces=[table_trace_index]  # Update the existing table trace
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Define slider steps
    slider_steps = []
    for i in range(len(windows)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows[i].index.min().strftime("%Y-%m")  # Label represents the start date of the window
        )
        slider_steps.append(step)
    
    # Optionally, reverse the slider steps to have newest on the left
    slider_steps = slider_steps[::-1]
    
    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]
    
    # Update layout with the slider
    fig.update_layout(
        sliders=sliders
    )
    
    # Add source annotation based on selected_period
    fig.add_annotation(
        text=("Source: " + source_links + " <br><a href='https://www.bls.gov/news.release/ppi.nr0.htm' style='color:white;' target='_blank'>BLS PPI Report (Weight needs to be updated manually)</a>"),
        font=dict(size=8, color="white"),
        xref="paper", yref="paper",
        x=0.5, y=-0.03,  # Position the source annotation at the bottom
        showarrow=False
    )
    
    # Update layout with dynamic title and other aesthetics
    fig.update_layout(
        title=title,  # Dynamic title based on MoM or YoY
        title_font=dict(size=8),
        width=1000,
        height=1000,  # Increased height for larger plots
        margin=dict(t=80, b=80, l=20, r=20),  # Adjusted side margins
        template='plotly_dark',
        barmode='relative',  # Enable stacked bars for the component plot
        showlegend=True,
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="top",    # Align to top
            y=1.2,            # Set the legend position slightly above the plots
            xanchor="center", # Center the legend
            x=0.5,
            font=dict(size=8)  # Set font size for the legend
        ),
        font=dict(size=8)  # Global font size for axis labels, tick marks, etc.
    )
    
    # Set fixed y-axis ranges based on selected_period
    fig.update_yaxes(range=[y1_min, y1_max], row=1, col=1)
    fig.update_yaxes(range=[y2_min, y2_max], row=2, col=1)
    
    return fig


# In[10]:


def generate_cpi_detailed_graph_9(contribution_type='MoM'):
    # Initialize Fred API with your API key
    fred = Fred(api_key='7227512392e5e5d2a2679a261d2bb3a9')  # Replace with your actual FRED API key
    
    # Fetch PPI component data from FRED
    series_ids = {
        'All Items PPI': 'PPIFID',        # All Items PPI
        'Core PPI': 'WPUFD413',           # Core PPI (Final Demand Goods Less Foods and Energy)
        'Foods PPI': 'PPIFDF',            # Foods PPI
        'Energy PPI': 'PPIFDE',           # Energy PPI
        'Goods PPI': 'PPIFDG',            # Goods PPI (Final Demand)
        'Services PPI': 'PPIFDS',         # Services PPI
        'Construction PPI': 'PPIFDC'      # Construction PPI
    }
    
    # Fetch data for each series from FRED and store in a DataFrame
    ppi_data = {name: fred.get_series(series_id) for name, series_id in series_ids.items()}
    df = pd.DataFrame(ppi_data)
    
    # Ensure the DataFrame index is in datetime format
    df.index = pd.to_datetime(df.index)
    
    # Define relative importance weights (these need to be manually updated based on the latest BLS release)
    relative_importance = {
        "Goods PPI": 33.2,         # Example weight, please update according to BLS data
        "Services PPI": 66.8,      # Example weight, please update according to BLS data
        "Foods PPI": 10.0,         # Example weight, please update according to BLS data
        "Energy PPI": 5.4,         # Example weight, please update according to BLS data
        "Construction PPI": 7.8,   # Example weight, please update according to BLS data
        "All Items PPI": 100.0     # Already includes Foods, Energy, etc.
    }
    
    # Determine the required buffer period and window size based on contribution_type
    if contribution_type == 'MoM':
        buffer_period = 1  # To compute MoM for the first display month
        window_size = 30   # Number of months in each window
        y1_min, y1_max = -3, 3      # Fixed y-axis range for Plot 1
    elif contribution_type == 'YoY':
        buffer_period = 12  # To compute YoY for the first display month
        window_size = 30    # Number of months in each window
        y1_min, y1_max = -3, 3      # Fixed y-axis range for Plot 1
    else:
        raise ValueError("contribution_type must be either 'MoM' or 'YoY'.")
    
    # Calculate the start date needed to accommodate the buffer
    display_start_date = datetime(2015, 1, 1)
    buffer_start_date = display_start_date - relativedelta(months=buffer_period)
    
    # Fetch data starting from buffer_start_date to ensure calculations are possible
    df = df[df.index >= buffer_start_date.strftime('%Y-%m-%d')]
    
    # Calculate MoM or YoY percentage changes based on user input
    if contribution_type == 'MoM':
        data_df = df.pct_change(periods=1) * 100
        title = "Current Graph Displays Month-over-Month (MoM) Changes"

        # MoM source links
        source_links = (
            "<a href='https://fred.stlouisfed.org/graph/?g=1tEWh' style='color:white'>All Items PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tEWu' style='color:white'>Core PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tEWx' style='color:white'>Foods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tEWD' style='color:white'>Energy PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tEWL' style='color:white'>Services PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF5t' style='color:white'>Goods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF5u' style='color:white'>Construction PPI</a>"
        )

    else:  # 'YoY'
        data_df = df.pct_change(periods=12) * 100
        title = "Current Graph Displays Year-over-Year (YoY) Changes"

        # Use MoM data to calculate the 6-month annualized growth rate
        data_df_mom = df.pct_change(periods=1)

        # Calculate the 6-month annualized rate using MoM data
        data_df['All Items PPI (6-mo Annualized)'] = (
            data_df_mom['All Items PPI'].rolling(window=6).apply(lambda x: np.prod(1 + x/100) ** (12 / 6) - 1)) * 100
        data_df['Core PPI (6-mo Annualized)'] = (
            data_df_mom['Core PPI'].rolling(window=6).apply(lambda x: np.prod(1 + x/100) ** (12 / 6) - 1)) * 100

        # YoY source links
        source_links = (
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF4M' style='color:white'>All Items PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF4P' style='color:white'>Core PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF4X' style='color:white'>Foods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF4D' style='color:white'>Energy PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF5t' style='color:white'>Services PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF5u' style='color:white'>Goods PPI</a>, "
            "<a href='https://fred.stlouisfed.org/graph/?g=1tF5v' style='color:white'>Construction PPI</a>"
        )

    # Drop rows with NaN values after calculations
    data_df = data_df.dropna()

    # Filter data from display_start_date onwards
    data_df = data_df[data_df.index >= display_start_date.strftime('%Y-%m-%d')]

    # Calculate contribution by multiplying percentage changes by relative importance
    contribution_df = data_df.copy()
    for column in ['Energy PPI', 'Foods PPI', 'Goods PPI', 'Services PPI', 'Construction PPI', 'All Items PPI']:
        contribution_df[column] = contribution_df[column] * relative_importance[column] / 100

    # Round the contribution data to 3 decimal places for plotting
    contribution_df = contribution_df.round(2)

    # Create subplots: main plot and table
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # Allocate more space for the plot
        subplot_titles=("", ""),
        vertical_spacing=0.05,
        specs=[[{"type": "xy"}], [{"type": "table"}]]
    )

    # Plot 1: Lines for All Items PPI and Core PPI
    fig.add_trace(go.Scatter(
        x=contribution_df.index, y=contribution_df['All Items PPI'],
        mode='lines', name='All Items PPI',
        line=dict(color='white')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=contribution_df.index, y=contribution_df['Core PPI'],
        mode='lines', name='Core PPI',
        line=dict(color='magenta', dash='dot')
    ), row=1, col=1)

    # Add 6-month annualized rate (for YoY plot)
    if contribution_type == 'YoY':
        fig.add_trace(go.Scatter(
            x=contribution_df.index, y=contribution_df['All Items PPI (6-mo Annualized)'],
            mode='lines', name='All Items PPI (6-mo Annualized)',
            line=dict(color='white', dash='dot')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=contribution_df.index, y=contribution_df['Core PPI (6-mo Annualized)'],
            mode='lines', name='Core PPI (6-mo Annualized)',
            line=dict(color='magenta', dash='dot')
        ), row=1, col=1)

    # Plot 1: Stacked Bar Plot for Contributions
    fig.add_trace(go.Bar(
        x=contribution_df.index, y=contribution_df['Services PPI'],
        name='Services PPI', marker_color='green'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_df.index, y=contribution_df['Goods PPI'],
        name='Goods PPI', marker_color='purple'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_df.index, y=contribution_df['Foods PPI'],
        name='Foods PPI', marker_color='blue'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_df.index, y=contribution_df['Energy PPI'],
        name='Energy PPI', marker_color='orange'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=contribution_df.index, y=contribution_df['Construction PPI'],
        name='Construction PPI', marker_color='yellow'
    ), row=1, col=1)

    # Ensure barmode is 'relative' for stacked bars
    fig.update_layout(barmode='relative')

    # Calculate the number of possible windows
    window_size = 30  # Number of months in each window for the slider
    data_length = len(contribution_df)
    max_start_index = data_length - window_size

    if max_start_index < 0:
        raise ValueError(f"Window size ({window_size}) is larger than the available data ({data_length} months).")

    # Generate all possible windows in chronological order
    windows = [contribution_df.iloc[i:i + window_size] for i in range(max_start_index + 1)]

    # Reverse the windows to have the most recent window first
    windows = windows[::-1]

    # Function to generate colors for each row based on values
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Prepare data for the initial table (most recent window)
    initial_window = windows[0]
    table_df = initial_window[['All Items PPI', 'Core PPI', 'Goods PPI', 'Services PPI', 'Foods PPI', 'Energy PPI', 'Construction PPI']].tail(window_size).round(3)
    table_df = table_df.T  # Transpose for desired format
    table_header_dates = table_df.columns.strftime("%y-%m")  # Format dates as 'YY-MM'

    # Generate colors for the initial table
    colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
    flat_colors = list(map(list, zip(*colors)))  # Transpose to match Plotly's requirement

    # Create the initial table
    table_trace = go.Table(
        header=dict(
            values=["Item", "Relative Importance (%)"] + list(table_header_dates),
            fill_color='black',
            font=dict(color='white', size=8),  # Set font size to 8
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                table_df.index,  # Row labels (Item names)
                [relative_importance.get(item, 0) for item in table_df.index],  # Relative Importance (%)
                *[table_df[col] for col in table_df.columns]  # Contribution values
            ],
            fill_color=[['black'] * len(table_header_dates), ['black'] * len(table_header_dates)] + flat_colors,
            font=dict(color='white', size=8),  # Set font size to 8
            align=['left', 'center'] + ['center'] * len(table_header_dates),
            height=20
        ),
        columnwidth=[4, 4] + [1] * len(table_header_dates)  # Adjust column widths
    )
    fig.add_trace(table_trace, row=2, col=1)  # Table is the second subplot

    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1  # Last trace added

    # Create frames for the slider
    frames = []
    for i, window in enumerate(windows):
        frame_table_df = window[['All Items PPI', 'Core PPI', 'Goods PPI', 'Services PPI', 'Foods PPI', 'Energy PPI', 'Construction PPI']].tail(window_size).round(3)
        frame_table_df = frame_table_df.T  # Transpose
        frame_header_dates = frame_table_df.columns.strftime("%y-%m")  # Format dates as 'YY-MM'

        # Generate colors for the table
        frame_colors = [get_row_heatmap_colors(row) for _, row in frame_table_df.iterrows()]
        frame_flat_colors = list(map(list, zip(*frame_colors)))  # Transpose

        # Define updated table cells
        updated_cells = dict(
            values=[
                frame_table_df.index,  # Row labels (Item names)
                [relative_importance.get(item, 0) for item in frame_table_df.index],  # Relative Importance (%)
                *[frame_table_df[col] for col in frame_table_df.columns]  # Contribution values
            ],
            fill_color=[['black'] * len(frame_header_dates), ['black'] * len(frame_header_dates)] + frame_flat_colors,
            font=dict(color='white', size=8),  # Set font size to 8
            align=['left', 'center'] + ['center'] * len(frame_header_dates),
            height=20
        )

        # Define updated table header
        updated_header = dict(
            values=["Item", "Relative Importance (%)"] + list(frame_header_dates),
            fill_color='black',
            font=dict(color='white', size=8),  # Set font size to 8
            align='center',
            height=20
        )

        # Calculate the start and end dates for the shaded window
        window_start = window.index.min()
        window_end = window.index.max()

        # Define the shaded rectangle shapes for Plot 1 with fixed y-axis ranges
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=y1_min,  # Shaded area starts at y1_min
            y1=y1_max,  # Shaded area ends at y1_max
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )

        # Create the frame with updated table and shaded shape
        frame = go.Frame(
            data=[
                go.Table(
                    header=updated_header,
                    cells=updated_cells
                )
            ],
            layout=go.Layout(
                shapes=[shaded_shape_plot1]  # Update the shaded area
            ),
            name=str(i),
            traces=[table_trace_index]  # Reference to the existing table trace
        )
        frames.append(frame)

    fig.frames = frames

    # Define slider steps
    slider_steps = []
    for i in range(len(windows)):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=windows[i].index.min().strftime("%y-%m")  # Label represents the start date of the window
        )
        slider_steps.append(step)

    # No need to reverse the slider_steps since we already reversed the windows
    # slider_steps = slider_steps[::-1]

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]

    # Add the initial shaded area corresponding to the first window (most recent)
    initial_window_start = windows[0].index.min()
    initial_window_end = windows[0].index.max()
    initial_shaded_shape = dict(
        type="rect",
        xref="x1",
        yref="y1",
        x0=initial_window_start,
        x1=initial_window_end,
        y0=y1_min,
        y1=y1_max,
        fillcolor="grey",
        opacity=0.2,
        line_width=0
    )
    fig.update_layout(shapes=[initial_shaded_shape])

    # Update layout with the slider
    fig.update_layout(
        sliders=sliders
    )

    # Add footnote with a link to BLS PPI report
    fig.add_annotation(
        text="Source: <a href='https://www.bls.gov/news.release/ppi.nr0.htm' style='color:white;' target='_blank'>BLS PPI Report (Weights need to be updated manually)</a>",
        showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.05,
        xanchor='center', yanchor='top',
        font=dict(size=8),
        align='center'
    )

    # Update layout to match the Bloomberg-style chart with left and right margins set to 20
    fig.update_layout(
        barmode='relative',
        height=900,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=8)
        ),
        font=dict(size=8),  # Global font size for axis labels, tick marks, etc.
        margin=dict(l=20, r=20, t=80, b=80)  # Adjusted side margins
    )

    # Fix the y-axis range to prevent dynamic scaling
    fig.update_yaxes(range=[y1_min, y1_max], row=1, col=1)  # Fixed y-axis range for Plot 1

    # Update Y-axis label to reflect that we're displaying Contributions (percentage points)
    fig.update_yaxes(title_text="Contributions (percentage points)", row=1, col=1)

    return fig



# In[11]:


def generate_cpi_detailed_graph_10():
    
    # Filter for CPI and Core CPI YoY data
    cpi_data = economic_calendar_data[economic_calendar_data['event'].str.contains("CPI", case=False)]
    core_cpi_yoy_data = economic_calendar_data[economic_calendar_data['event'].str.contains("Core CPI \(YoY\)", case=False)]

    core_cpi_yoy_data['date'] = pd.to_datetime(core_cpi_yoy_data['date'], format='%d/%m/%Y')
    
    # Ensure 'actual' and 'forecast' are strings
    core_cpi_yoy_data['actual'] = core_cpi_yoy_data['actual'].astype(str)
    core_cpi_yoy_data['forecast'] = core_cpi_yoy_data['forecast'].astype(str)
    
    # Handle missing or non-numeric values
    core_cpi_yoy_data['actual'] = core_cpi_yoy_data['actual'].str.replace('%', '')
    core_cpi_yoy_data['forecast'] = core_cpi_yoy_data['forecast'].str.replace('%', '')
    
    # Convert 'actual' and 'forecast' to float, handling non-numeric values gracefully
    core_cpi_yoy_data['actual'] = pd.to_numeric(core_cpi_yoy_data['actual'], errors='coerce') 
    core_cpi_yoy_data['forecast'] = pd.to_numeric(core_cpi_yoy_data['forecast'], errors='coerce') 
    
    # Drop rows with NaN values in 'actual' or 'forecast', except for the last row
    core_cpi_yoy_data.dropna(subset=['actual', 'forecast'], inplace=True)
    
    # Calculate the surprise element
    core_cpi_yoy_data['surprise'] = (core_cpi_yoy_data['actual'] - core_cpi_yoy_data['forecast']) / core_cpi_yoy_data['forecast']
    
    # Handle NaN in the last data point for 'surprise'
    if core_cpi_yoy_data['surprise'].isna().iloc[-1]:
        core_cpi_yoy_data = core_cpi_yoy_data.iloc[:-1]
    
    # Define the decay factor
    decay_factor = 1 / 6
    
    # Apply the decay to the surprise element
    core_cpi_yoy_data['smoothed_surprise'] = core_cpi_yoy_data['surprise'].ewm(span=6, adjust=False).mean()
    
    # Calculate the overall Economic Surprise Index
    economic_surprise_index = core_cpi_yoy_data['smoothed_surprise'].mean()
    print("Economic Surprise Index:", economic_surprise_index)
    
    # Create the figure with two subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('', 'Inflation Surprise Index'),
                        shared_xaxes=True,  # Share the x-axis between the two plots
                        vertical_spacing=0.02)
    
    # Plot Actual vs Forecast Core CPI YoY
    fig.add_trace(go.Scatter(
        x=core_cpi_yoy_data['date'],
        y=core_cpi_yoy_data['actual'],  # Keep as percentage for plotting
        mode='lines+markers',
        name='Actual Core CPI YoY',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=core_cpi_yoy_data['date'],
        y=core_cpi_yoy_data['forecast'],  # Keep as percentage for plotting
        mode='lines+markers',
        name='Forecast Core CPI YoY',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)
    
    # Plot Economic Surprise Index
    fig.add_trace(go.Scatter(
        x=core_cpi_yoy_data['date'],
        y=core_cpi_yoy_data['smoothed_surprise']*100,  # Convert to percentage for plotting
        mode='lines+markers',
        name='Smoothed Surprise',
        line=dict(color='green', dash='dot')
    ), row=2, col=1)
    
    # Add shaded areas for background
    for index, row in core_cpi_yoy_data.iterrows():
        if row['smoothed_surprise'] > 0:
            fig.add_shape(
                type='rect',
                x0=row['date'],
                x1=row['date'],
                y0=0,
                y1=row['smoothed_surprise']*100,
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(width=0),
                row=2, col=1
            )
        else:
            fig.add_shape(
                type='rect',
                x0=row['date'],
                x1=row['date'],
                y0=row['smoothed_surprise']*100,
                y1=0,
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(width=0),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.1  # Set the thickness of the range slider
            ),
            type='date'
        ),
        template='plotly_dark',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Core CPI YoY (%)", row=1, col=1)
    fig.update_yaxes(title_text="Surprise Index (%)", row=2, col=1)
    
    return fig



# In[12]:


def generate_cpi_detailed_graph_11():
    # Retrieve GDP deflator and CPI data from FRED
    gdp_deflator = fred.get_series('GDPDEF', observation_start='2000-01-01', observation_end='2026-01-01')
    cpi_qoq = fred.get_series('CPIAUCSL', observation_start='2000-01-01', observation_end='2026-01-01')

    # Align both datasets to avoid mismatches in dates
    combined_data = pd.concat([gdp_deflator, cpi_qoq], axis=1, join='inner')
    combined_data.columns = ['gdp_deflator', 'cpi_qoq']

    # Calculate the QoQ growth rates for GDP deflator and CPI
    combined_data['gdp_deflator_qoq'] = combined_data['gdp_deflator'].pct_change() * 100  # Convert to percentage
    combined_data['cpi_qoq_growth'] = combined_data['cpi_qoq'].pct_change() * 100  # Convert to percentage

    # Drop the initial rows that may have NaN values due to pct_change()
    combined_data.dropna(inplace=True)

    # Colormap functions for heatmap colors
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Create subplots for graph and table
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "xy", "secondary_y": True}], [{"type": "table"}]]
    )

    # Plot all data starting from 2000 in the graph
    fig.add_trace(go.Bar(
        x=combined_data.index, y=combined_data['gdp_deflator_qoq'],
        name='GDP Deflator QoQ',
        marker=dict(color='white')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=combined_data.index, y=combined_data['cpi_qoq_growth'],
        mode='lines', name='CPI QoQ',
        line=dict(color='cyan')
    ), row=1, col=1, secondary_y=True)

    # Prepare all the windows for the slider (only affects the table)
    window_size = 30
    data_length = len(combined_data)
    max_start_index = data_length - window_size
    windows = [combined_data.iloc[i:i + window_size] for i in range(max_start_index + 1)]

    # Reverse the windows to make new data appear on the left and old on the right
    windows = windows[::-1]

    # Function to create table based on the selected window
    def create_table(data_window):
        table_data = pd.DataFrame({
            'Metric': ['GDP Deflator (%)', 'CPI (%)'],
        })

        # Add the actual values for the selected window
        for date in data_window.index:
            table_data[date.strftime('%y-%m')] = [
                data_window.loc[date]['gdp_deflator'].round(1),
                data_window.loc[date]['cpi_qoq'].round(1)
            ]

        # Generate colors for the table based on the data rows
        colors = [get_row_heatmap_colors(row) for _, row in table_data.iloc[:, 1:].iterrows()]
        flat_colors = list(map(list, zip(*colors)))  # Transpose the colors to match Plotly's column-wise format

        # Add the table trace with custom formatting
        table_trace = go.Table(
            header=dict(
                values=["Metric"] + [date.strftime('%y-%m') for date in data_window.index],
                fill_color='black',
                font=dict(color='white', size=8),
                align='center'
            ),
            cells=dict(
                values=[table_data['Metric']] + [table_data[date.strftime('%y-%m')].astype(str) for date in data_window.index],
                fill_color=[['black'] * len(data_window.index)] + flat_colors,  # Apply the heatmap colors
                font=dict(color='white', size=8),
                align='center',
                height=20
            )
        )
        return table_trace

    # Add the initial table
    initial_window = windows[-1]  # Show the most recent 30 months initially
    table_trace = create_table(initial_window)
    fig.add_trace(table_trace, row=2, col=1)

    # Add slider functionality (affecting only the table)
    frames = []
    for i, window in enumerate(windows):
        frame_table = create_table(window)

        # Create frame with updated table and shaded area on the graph
        window_start = window.index.min()
        window_end = window.index.max()

        # Define the grey shaded area for the 30-month window in the main plot
        shaded_shape = dict(
            type="rect",
            xref="x",  # Reference the x-axis of the main plot
            yref="paper",  # Reference the paper coordinates for the height of the shading
            x0=window_start,
            x1=window_end,
            y0=0,  # Shaded area starts from the bottom of the plot
            y1=1,  # Shaded area goes to the top of the plot
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )

        # Create frame with updated table and shaded area
        frame = go.Frame(
            data=[frame_table],
            layout=go.Layout(shapes=[shaded_shape]),
            name=str(i),
            traces=[2]  # Reference the table trace index (traces start from 0, so table is trace 2)
        )
        frames.append(frame)

    # Add frames to figure
    fig.frames = frames

    # Define slider steps (for table only) - reverse order for left-to-right new-to-old display
    slider_steps = []
    for i, window in enumerate(windows):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
            ],
            label=window.index.min().strftime('%y-%m')
        )
        slider_steps.append(step)

    # Add slider to layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps
    )]

    # Update layout with the slider
    fig.update_layout(
        title={
            'text': 'U.S. GDP Deflator and CPI (QoQ)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='GDP Deflator QoQ (%)',
        yaxis2=dict(
            title='CPI QoQ (%)',
            overlaying='y',
            side='right',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
            range=[-2, 2]
        ),
        sliders=sliders,
        barmode='stack',
        template='plotly_dark',
        font=dict(color='white'),
        width=1000,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        annotations=[
            dict(
                x=0.5,
                y=-0.015,
                showarrow=False,
                text="Federal Reserve Economic Data (FRED) |<a href='https://fred.stlouisfed.org/graph/?g=1tS31' style='color:white'>Gross Domestic Product: Implicit Price Deflator</a>",
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='top',
                font=dict(color='white')
            )
        ]
    )

    return fig


# In[13]:


def generate_cpi_detailed_graph_12():
    # Initialize FRED with the provided API Key
    fred = Fred(api_key=api_key)

    # Retrieve the expected inflation data for 1-year, 2-year, 3-year, and 5-year
    expinf1yr = fred.get_series('EXPINF1YR')
    expinf2yr = fred.get_series('EXPINF2YR')
    expinf3yr = fred.get_series('EXPINF3YR')
    expinf5yr = fred.get_series('EXPINF5YR')

    # Filter data from 2015 onwards
    start_date = '2015-01-01'
    expinf1yr = expinf1yr[expinf1yr.index >= start_date]
    expinf2yr = expinf2yr[expinf2yr.index >= start_date]
    expinf3yr = expinf3yr[expinf3yr.index >= start_date]
    expinf5yr = expinf5yr[expinf5yr.index >= start_date]

    # Prepare date range for plotting
    dates = expinf1yr.index
    window_size = 30  # Number of months to display in the table and shading window

    # Function to generate colormap for table cells (green for lower values, red for higher)
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()  # Get min and max values for normalization
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1

        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Create subplots: graph on top, table below
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],  # 75% for graph, 25% for table
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
        vertical_spacing=0.08  # Increase gap between the graph and the table
    )

    # Add the line plot on top
    fig.add_trace(go.Scatter(
        x=dates, y=expinf1yr, mode='lines', name='1-Year Expected Inflation',
        line=dict(color='cyan', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=expinf2yr, mode='lines', name='2-Year Expected Inflation',
        line=dict(color='yellow', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=expinf3yr, mode='lines', name='3-Year Expected Inflation',
        line=dict(color='green', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=expinf5yr, mode='lines', name='5-Year Expected Inflation',
        line=dict(color='red', width=2)
    ), row=1, col=1)

    # Add 2% Fed target line (dashed) for all expected inflation plots
    fig.add_trace(go.Scatter(
        x=dates, y=[2]*len(dates), mode='lines', name='2% Fed Target',
        line=dict(color='white', dash='dash'),  # Dotted white line
        showlegend=True
    ), row=1, col=1)

    # Create the initial table below the graph with the most recent data
    initial_table_data = pd.DataFrame({
        "1-Year Exp. Inflation (%)": expinf1yr.tail(30).values,
        "2-Year Exp. Inflation (%)": expinf2yr.tail(30).values,
        "3-Year Exp. Inflation (%)": expinf3yr.tail(30).values,
        "5-Year Exp. Inflation (%)": expinf5yr.tail(30).values
    }).T.round(2)

    initial_dates_for_table = expinf1yr.tail(30).index.strftime("%y-%m")

    # Generate colors for the initial table using the green-to-red colormap function
    colors = [get_row_heatmap_colors(row) for _, row in initial_table_data.iterrows()]
    flat_colors = list(map(list, zip(*colors)))  # Transpose colors to match Plotly's format

    # Add the initial table to the figure
    fig.add_trace(go.Table(
        header=dict(
            values=["Items"] + list(initial_dates_for_table),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        cells=dict(
            values=[
                initial_table_data.index,  # Row labels
                *[initial_table_data[col] for col in initial_table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(initial_dates_for_table)] + flat_colors,
            font=dict(color='white', size=8),
            align='center',
            height=20
        ),
        columnwidth=[4] + [0.5] * len(initial_dates_for_table)  # Adjust column widths
    ), row=2, col=1)

    # Identify the trace index of the table
    table_trace_index = len(fig.data) - 1

    # Create frames for table data based on different windows
    frames = []
    for i in range(len(dates) - window_size):
        window_start = dates[i]
        window_end = dates[i + window_size]

        # Create table data for each window
        window_data = {
            "1-Year Exp. Inflation (%)": expinf1yr[i:i+30].values,
            "2-Year Exp. Inflation (%)": expinf2yr[i:i+30].values,
            "3-Year Exp. Inflation (%)": expinf3yr[i:i+30].values,
            "5-Year Exp. Inflation (%)": expinf5yr[i:i+30].values
        }
        table_data = pd.DataFrame(window_data).T.round(2)
        table_dates = expinf1yr[i:i+30].index.strftime("%y-%m")

        # Generate heatmap colors for the current table window
        window_colors = [get_row_heatmap_colors(row) for _, row in table_data.iterrows()]
        flat_colors = list(map(list, zip(*window_colors)))  # Transpose colors

        # Create updated header and cells for the table
        updated_header = dict(
            values=["Items"] + list(table_dates),
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=20
        )

        updated_cells = dict(
            values=[
                table_data.index,  # Row labels
                *[table_data[col] for col in table_data.columns]  # Column data
            ],
            fill_color=[['black'] * len(table_dates)] + flat_colors,
            font=dict(color='white', size=8),
            align='center',
            height=20
        )

        # Define shaded areas for the main plot
        shaded_shape_plot1 = dict(
            type="rect",
            xref="x1",  # References the x-axis of the first subplot
            yref="y1",  # References the y-axis of the first subplot only
            x0=window_start,
            x1=window_end,
            y0=-0.5,
            y1=5,  # Customizable range of the y-axis
            fillcolor="grey",
            opacity=0.2,
            line_width=0
        )

        # Create a frame with updated table data and shaded area
        frame = go.Frame(
            data=[
                go.Table(header=updated_header, cells=updated_cells)
            ],
            layout=go.Layout(shapes=[shaded_shape_plot1]),
            name=str(i),
            traces=[table_trace_index]
        )
        frames.append(frame)

    fig.frames = frames

    # Create slider steps for the animation (Newest to Oldest)
    slider_steps = []
    for i in range(len(frames)-1, -1, -1):  # Reverse the slider steps
        step = dict(
            method="animate",
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=dates[i].strftime("%Y-%m")
        )
        slider_steps.append(step)

    # Add slider that controls the table
    fig.update_layout(
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Start Date: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }]
    )

    # Add source note with FRED links
    fig.add_annotation(
        text=('Source: Federal Reserve Economic Data (FRED). '
              '<a href="https://fred.stlouisfed.org/series/EXPINF1YR" style="color: cyan">1-Year Expected Inflation</a>, '
              '<a href="https://fred.stlouisfed.org/series/EXPINF2YR" style="color: cyan">2-Year Expected Inflation</a>, '
              '<a href="https://fred.stlouisfed.org/series/EXPINF3YR" style="color: cyan">3-Year Expected Inflation</a>, '
              '<a href="https://fred.stlouisfed.org/series/EXPINF5YR" style="color: cyan">5-Year Expected Inflation</a>.'),
        font=dict(size=8, color="white"),
        xref="paper", yref="paper",
        x=0.5, y=-0.1,  # Position below the plot
        showarrow=False
    )

    # Update layout settings
    fig.update_layout(
        template="plotly_dark",
        font=dict(color="white", size=10),
        title="Expected Inflation for 1, 2, 3, and 5 Years",
        height=900,
        width=1000,
        margin=dict(t=50, b=150, l=20, r=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


# ## Tables

# In[14]:


def generate_cpi_detailed_table_1():

    # Initialize FRED API
    fred = Fred(api_key=api_key)

    # Define the series IDs for the required inflation data from FRED
    series_ids = {
        'PPI': 'PPIACO',                             # Producer Price Index for All Commodities
        'PCE': 'PCEPI',                              # Personal Consumption Expenditures Price Index
        'Cleveland Median CPI': 'MEDCPIM158SFRBCLE', # Cleveland Median CPI
        '16% Trimmed Mean CPI': 'TRMMEANCPIM158SFRBCLE', # 16% Trimmed Mean CPI
        'Sticky CPI': 'STICKCPIM157SFRBATL',         # Atlanta Fed Sticky CPI
        'Trimmed Mean PCE': 'PCETRIM1M158SFRBDAL'    # Dallas Fed Trimmed Mean PCE  
    }

    # Determine the buffer period for MoM calculations
    buffer_period = 1  # 1 month buffer for MoM (December 2014 needed for January 2015)

    # Calculate the start date needed to accommodate the buffer
    display_start_date = datetime(2015, 1, 1)
    buffer_start_date = display_start_date - relativedelta(months=buffer_period)

    # Fetch and process data for each series
    data = {}
    for name, series_id in series_ids.items():
        try:
            # Fetch the series data
            series_data = fred.get_series(series_id)
            # Ensure the index is datetime
            series_data.index = pd.to_datetime(series_data.index)
            # Filter data from buffer_start_date onwards
            series_data = series_data[series_data.index >= buffer_start_date]
            # Calculate MoM percentage changes
            series_pct_change = series_data.pct_change(periods=1) * 100
            data[name] = series_pct_change
        except Exception as e:
            print(f"Error fetching series {name} with ID {series_id}: {e}")
            data[name] = pd.Series(dtype=float)

    # Create a DataFrame from the retrieved data
    df = pd.DataFrame(data)

    # Drop rows with all NaN values to clean up the DataFrame
    df.dropna(how='all', inplace=True)

    # Filter to show only data from 2015-01-01 onwards
    df = df[df.index >= display_start_date]

    # Fill NA values to avoid issues in the table
    df.fillna(0, inplace=True)

    # Convert dates to 'YY-MM' format for the table
    df.index = pd.to_datetime(df.index).strftime('%y-%m')

    # Determine the total number of months available
    total_months = len(df)
    window_size = 30  # Number of months in each window
    step_size = 6     # Step size in months

    # Generate all possible windows with 6-month steps
    windows = []
    window_labels = []
    for start in range(0, total_months - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        windows.append(window)
        window_labels.append(window.index[0])  # Label with the start date of the window

    # Reverse the windows and labels to have newest on the left
    windows = windows[::-1]
    window_labels = window_labels[::-1]

    # Function to generate colors for heatmap per item
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Create the initial table data (first window)
    initial_window = windows[0]
    table_df = initial_window.transpose().round(2)
    table_header_dates = table_df.columns.tolist()  # Corrected: Use columns (dates) for headers

    # Generate colors for the initial table
    colors = [get_row_heatmap_colors(row) for _, row in table_df.iterrows()]
    flat_colors = list(map(list, zip(*colors)))  # Transpose to match Plotly's requirement

    # Prepare the colors for Plotly Table
    color_fill = [['black'] * len(table_header_dates)] + flat_colors  # Header colors + Cell colors

    # Modify the columnwidth to ensure the first column is wider
    columnwidth = [4] + [1] * len(table_header_dates)

    # Create the initial table
    fig = go.Figure(
        data=[go.Table(
            columnwidth=columnwidth,  # Adjusted columnwidth
            header=dict(
                values=["Item"] + list(table_header_dates),
                fill_color='black',
                font=dict(color='white', size=8),  # Font size for the header
                align='center',
                height=30
            ),
            cells=dict(
                values=[table_df.index] + [table_df[col] for col in table_df.columns],
                fill=dict(color=color_fill),
                font=dict(color='white', size=8),  # Font size for the cells
                align=['left'] + ['center'] * len(table_header_dates),
                height=20  # Cell height for better visibility
            )
        )]
    )

    # Add frames for each window
    frames = []
    for i, window in enumerate(windows):
        frame_table_df = window.transpose().round(2)
        frame_header_dates = frame_table_df.columns.tolist()  # Corrected: Use columns (dates) for headers

        # Generate colors for the table
        frame_colors = [get_row_heatmap_colors(row) for _, row in frame_table_df.iterrows()]
        frame_flat_colors = list(map(list, zip(*frame_colors)))  # Transpose

        # Prepare fill colors
        frame_color_fill = [['black'] * len(frame_header_dates)] + frame_flat_colors  # Header + Cells

        # Define the table for this frame
        frame_table = go.Table(
            columnwidth=columnwidth,  # Ensure column widths are consistent
            header=dict(
                values=["Item"] + list(frame_header_dates),
                fill_color='black',
                font=dict(color='white', size=8),
                align='center',
                height=30
            ),
            cells=dict(
                values=[frame_table_df.index] + [frame_table_df[col] for col in frame_table_df.columns],
                fill=dict(color=frame_color_fill),
                font=dict(color='white', size=8),
                align=['left'] + ['center'] * len(frame_header_dates),
                height=20
            )
        )

        # Create the frame
        frame = go.Frame(
            data=[frame_table],
            name=str(i)  # Frame name must be a string
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Define slider steps
    slider_steps = []
    for i, label in enumerate(window_labels):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=label  # Label represents the start date of the window
        )
        slider_steps.append(step)

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps,
        len=0.9,
        x=0.05,
        y=0
    )]

    # Update layout with the slider
    fig.update_layout(
        sliders=sliders
    )

    # Add source links as annotations below the table
    source_text = (
        "Sources: "
        "<a href='https://fred.stlouisfed.org/graph/?g=1uO7f' style='color:white'>PPI</a>, "
        "<a href='https://fred.stlouisfed.org/graph/?g=1uO8j' style='color:white'>Median Consumer Price Index</a>"
    )

    fig.add_annotation(
        text=source_text,
        showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.2,  # Positioned below the table
        xanchor='center', yanchor='top',
        font=dict(size=8),
        align='center'
    )

    # Update layout for a black background
    fig.update_layout(
        width=1000,  # Adjust width to fit the table
        height=300,  # Adjust height for better visibility
        template='plotly_dark',  # Use Plotly's dark template
        margin=dict(l=20, r=20, t=20, b=50),
        plot_bgcolor='black',  # Set the plot background color to black
        paper_bgcolor='black'  # Set the paper background color to black
    )

    return fig


# In[15]:


def generate_cpi_detailed_table_2(): 
    def process_event_data(event_name):
        # Filter event-related data (MoM events for CPI, PCE, PPI, Cleveland CPI)
        event_data = economic_calendar_data[economic_calendar_data['event'].str.contains(f"{event_name} \\(MoM\\)", case=False)]
        
        # Convert 'date' to datetime
        event_data['date'] = pd.to_datetime(event_data['date'], format='%d/%m/%Y')

        # Ensure 'actual' and 'forecast' are strings
        event_data['actual'] = event_data['actual'].astype(str)
        event_data['forecast'] = event_data['forecast'].astype(str)

        # Handle missing or non-numeric values
        event_data['actual'] = event_data['actual'].str.replace('%', '')
        event_data['forecast'] = event_data['forecast'].str.replace('%', '')

        # Convert 'actual' and 'forecast' to float, handling non-numeric values
        event_data['actual'] = pd.to_numeric(event_data['actual'], errors='coerce')
        event_data['forecast'] = pd.to_numeric(event_data['forecast'], errors='coerce')

        # Drop rows with NaN values in 'actual' or 'forecast'
        event_data.dropna(subset=['actual', 'forecast'], inplace=True)

        # Keep only the most recent event data for each month
        event_data['year_month'] = event_data['date'].dt.to_period('M')
        event_data = event_data.groupby('year_month').apply(lambda x: x.loc[x['date'].idxmax()]).reset_index(drop=True)

        # Calculate the surprise element
        event_data['surprise'] = (event_data['actual'] - event_data['forecast']) / event_data['forecast'] * 100

        # Select the last 30 periods (months)
        last_30_periods = event_data.tail(30)

        # Create a DataFrame for the table
        df_table = last_30_periods[['date', 'actual', 'forecast', 'surprise']].copy()
        df_table.columns = ['Date', 'Actual (%)', 'Forecast (%)', 'Surprise (%)']

        # Format the date to show only year and month
        df_table['Date'] = df_table['Date'].dt.strftime('%y-%m')

        # Fill NaN values with 0 to avoid invalid color generation
        df_table.fillna(0, inplace=True)

        # Transpose the DataFrame to have periods as columns and metrics as rows
        df_table_transposed = df_table.set_index('Date').transpose()

        return df_table_transposed

    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    def get_heatmap_color(value, vmin, vmax):
        if np.isnan(value):
            return 'rgba(100, 100, 100, 0.6)'  # Grey for NaN values
        norm_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0
        if value < 0:
            return f'rgba(0, {int(255 * (1 + norm_value))}, 0, 0.6)'  # Green for deflation
        else:
            return f'rgba({int(255 * norm_value)}, 0, 0, 0.6)'  # Red for inflation

    def create_color_fill_per_component(df):
        color_fill = []
        for row in df.values:
            row_colors = get_row_heatmap_colors(row)
            color_fill.append(row_colors)
        color_fill_transposed = list(zip(*color_fill))
        return color_fill_transposed

    # Process Core CPI, Cleveland CPI, PCE, and PPI data
    cpi_table = process_event_data("Core CPI")
    pce_table = process_event_data("PCE Price Index")
    ppi_table = process_event_data("PPI")

    # Create subplots with 3 rows, reduced vertical spacing
    fig = make_subplots(
        rows=3, cols=1,
        specs=[[{'type': 'table'}], [{'type': 'table'}], [{'type': 'table'}]],
        subplot_titles=("Core CPI (MoM)", "PCE Price Index (MoM)", "PPI (MoM)"),
        vertical_spacing=0.01  # Reduced vertical space between plots
    )

    # Add Core CPI table
    if not cpi_table.empty:
        color_fill_cpi = create_color_fill_per_component(cpi_table)
        fig.add_trace(go.Table(
            columnwidth=[150] + [80] * len(cpi_table.columns),
            header=dict(values=["Item"] + list(cpi_table.columns), fill_color='black', font=dict(color='white', size=8), align='center'),
            cells=dict(values=[cpi_table.index] + [cpi_table[col].round(2) for col in cpi_table.columns],
                       fill=dict(color=[['black'] * len(cpi_table.columns)] + color_fill_cpi), font=dict(color='white', size=8),
                       align='center', height=20)),
            row=1, col=1
        )

    # Add PCE table
    if not pce_table.empty:
        color_fill_pce = create_color_fill_per_component(pce_table)
        fig.add_trace(go.Table(
            columnwidth=[150] + [80] * len(pce_table.columns),
            header=dict(values=["Item"] + list(pce_table.columns), fill_color='black', font=dict(color='white', size=8), align='center'),
            cells=dict(values=[pce_table.index] + [pce_table[col].round(2) for col in pce_table.columns],
                       fill=dict(color=[['black'] * len(pce_table.columns)] + color_fill_pce), font=dict(color='white', size=8),
                       align='center', height=20)),
            row=2, col=1
        )

    # Add PPI table
    if not ppi_table.empty:
        color_fill_ppi = create_color_fill_per_component(ppi_table)
        fig.add_trace(go.Table(
            columnwidth=[150] + [80] * len(ppi_table.columns),
            header=dict(values=["Item"] + list(ppi_table.columns), fill_color='black', font=dict(color='white', size=8), align='center'),
            cells=dict(values=[ppi_table.index] + [ppi_table[col].round(2) for col in ppi_table.columns],
                       fill=dict(color=[['black'] * len(ppi_table.columns)] + color_fill_ppi), font=dict(color='white', size=8),
                       align='center', height=20)),
            row=3, col=1
        )

    # Single source note for CPI, PCE, and PPI
    annotations = [
        dict(
            text=("Source: Economic Calendar Data from Investing.com | "
                  "<a href='https://www.investing.com/economic-calendar/cpi-733'>CPI</a>, "
                  "<a href='https://www.investing.com/economic-calendar/pce-price-index-735'>PCE</a>, "
                  "<a href='https://www.investing.com/economic-calendar/ppi-734'>PPI</a> | "
                  "Note: MoM Growth"),
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            font=dict(size=8, color="white"),
            align="left"
        )
    ]

    # Update layout for a black background, add the main title and annotations
    fig.update_layout(
        title="Economic Indicators Comparison: CPI, PCE, and PPI",
        width=1000,
        height=600,  # Increased height to accommodate more rows
        template='plotly_dark',
        margin=dict(l=20, r=20, t=80, b=80),
        annotations=annotations
    )

    return fig


# In[16]:


def generate_cpi_detailed_summary_table_1(observation_start='2014-12-01'):
    # Initialize FRED API
    fred = Fred(api_key=api_key)

    # Define the series IDs for the required inflation data from FRED
    series_ids = {
        'All Items': 'CPIAUCSL',
        'Food and Beverages': 'CPIFABSL',
        'Housing': 'CPIHOSSL',
        'Apparel': 'CPIAPPSL',
        'Transportation': 'CPITRNSL',
        'Medical Care': 'CPIMEDSL',
        'Recreation': 'CPIRECSL',
        'Education and Communication': 'CPIEDUSL',
        'Other Goods and Services': 'CPIOGSSL',
        'Core CPI': 'CPILFESL',
        '  Commodities': 'CUSR0000SAC',
        '  Energy': 'CPIENGSL',
        '  Services': 'CUSR0000SAS',
        '  Non Durables': 'CUSR0000SAN',
        'All Items less Food': 'CPIULFSL',
        'All Items less Energy': 'CPILEGSL',
        'All Items less Food and Energy': 'CPILFESL'
    }

    bold_categories = [
        'All Items', 'Core CPI', 'All Items less Food',
        'All Items less Energy', 'All Items less Food and Energy'
    ]

    # Fetch data from FRED and calculate MoM and YoY rates
    data_mom = {}
    data_yoy = {}
    for name, series_id in series_ids.items():
        while True:
            try:
                # Fetch data from FRED
                series_data = fred.get_series(series_id, observation_start=observation_start).resample('M').mean()
                break
            except ValueError as e:
                if "Too Many Requests" in str(e):
                    print("Rate limit exceeded. Waiting before retrying...")
                    time.sleep(60)  # Wait for a minute before retrying
                else:
                    print(f"Error fetching series {name} with ID {series_id}: {e}")
                    data_mom[name] = pd.Series(dtype=float)
                    data_yoy[name] = pd.Series(dtype=float)
                    break

        if not series_data.empty:
            # Calculate MoM and YoY rates
            data_mom[name] = series_data.pct_change() * 100  # MoM percentage change
            data_yoy[name] = (series_data / series_data.shift(12) - 1) * 100  # YoY percentage change

    # Create DataFrames for MoM and YoY
    df_mom = pd.DataFrame(data_mom).dropna(how='all')
    df_yoy = pd.DataFrame(data_yoy).dropna(how='all')

    # Filter to show only data from January 2015 onwards
    display_start_date = datetime(2015, 1, 1)
    df_mom = df_mom[df_mom.index >= display_start_date]
    df_yoy = df_yoy[df_yoy.index >= display_start_date]

    # Fill NA values to avoid issues in the table
    df_mom.fillna(0, inplace=True)
    df_yoy.fillna(0, inplace=True)

    # Convert dates to 'YY-MM' format for the table
    df_mom.index = pd.to_datetime(df_mom.index).strftime('%y-%m')
    df_yoy.index = pd.to_datetime(df_yoy.index).strftime('%y-%m')

    # Define window parameters
    window_size = 15  # Number of months in each window
    step_size = 4     # Step size in months

    # Generate all possible windows with 4-month steps
    def generate_windows(df, window_size, step_size):
        total_months = len(df)
        windows = []
        window_labels = []
        for start in range(0, total_months - window_size + 1, step_size):
            window = df.iloc[start:start + window_size]
            windows.append(window)
            window_labels.append(window.index[0])  # Label with the start date of the window
        # Reverse to have newest on the left
        windows = windows[::-1]
        window_labels = window_labels[::-1]
        return windows, window_labels

    windows_mom, window_labels_mom = generate_windows(df_mom, window_size, step_size)
    windows_yoy, window_labels_yoy = generate_windows(df_yoy, window_size, step_size)

    # Ensure both MoM and YoY have the same number of windows
    min_windows = min(len(windows_mom), len(windows_yoy))
    windows_mom = windows_mom[:min_windows]
    windows_yoy = windows_yoy[:min_windows]
    window_labels = window_labels_mom[:min_windows]  # Use MoM labels

    # Function to generate colors for heatmap per row
    def get_row_heatmap_colors(row):
        vmin, vmax = row.min(), row.max()
        return [get_heatmap_color(value, vmin, vmax) for value in row]

    # Function to get a heatmap color for a given value
    def get_heatmap_color(value, vmin, vmax):
        if vmax - vmin == 0:
            norm_value = 0.5  # Avoid division by zero
        else:
            norm_value = (value - vmin) / (vmax - vmin)  # Normalize between 0 and 1
        if value < 0:
            # Green shades for negative values (deflation)
            green_intensity = int(255 * (1 + norm_value))  # Scale between 0 and 255
            return f'rgba(0, {green_intensity}, 0, 0.6)'
        else:
            # Red shades for positive values (inflation)
            red_intensity = int(255 * norm_value)  # Scale between 0 and 255
            return f'rgba({red_intensity}, 0, 0, 0.6)'

    # Create the initial MoM table (first window)
    initial_window_mom = windows_mom[0]
    table_df_mom = initial_window_mom.transpose().round(1)
    table_header_dates_mom = table_df_mom.columns.tolist()  # Dates as 'YY-MM'

    # Generate colors for the initial MoM table
    colors_mom = [get_row_heatmap_colors(row) for _, row in table_df_mom.iterrows()]
    flat_colors_mom = list(map(list, zip(*colors_mom)))  # Transpose to match Plotly's requirement

    # Prepare the colors for Plotly Table (MoM)
    color_fill_mom = [['black'] * len(table_header_dates_mom)] + flat_colors_mom  # Header + Cells

    # Format categories with bold and indentation
    formatted_categories_mom = []
    for category in table_df_mom.index:
        if category in bold_categories:
            formatted_categories_mom.append(f"<b>{category.strip()}</b>")
        elif category.startswith('  '):  # Indent subcategories
            formatted_categories_mom.append(f"&nbsp;&nbsp;&nbsp;{category.strip()}")
        else:
            formatted_categories_mom.append(category)

    # Create the initial MoM table
    fig = go.Figure()

    fig.add_trace(go.Table(
        name='MoM',
        columnwidth=[200] + [80 for _ in range(len(table_header_dates_mom))],
        header=dict(
            values=["Category"] + table_header_dates_mom,
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=30
        ),
        cells=dict(
            values=[formatted_categories_mom] + [table_df_mom[col] for col in table_df_mom.columns],
            fill=dict(color=color_fill_mom),
            font=dict(color='white', size=8),
            align=['left'] + ['center'] * len(table_header_dates_mom),
            height=20
        ),
        visible=True
    ))

    # Create the initial YoY table (first window)
    initial_window_yoy = windows_yoy[0]
    table_df_yoy = initial_window_yoy.transpose().round(1)
    table_header_dates_yoy = table_df_yoy.columns.tolist()  # Dates as 'YY-MM'

    # Generate colors for the initial YoY table
    colors_yoy = [get_row_heatmap_colors(row) for _, row in table_df_yoy.iterrows()]
    flat_colors_yoy = list(map(list, zip(*colors_yoy)))  # Transpose to match Plotly's requirement

    # Prepare the colors for Plotly Table (YoY)
    color_fill_yoy = [['black'] * len(table_header_dates_yoy)] + flat_colors_yoy  # Header + Cells

    # Format categories with bold and indentation
    formatted_categories_yoy = []
    for category in table_df_yoy.index:
        if category in bold_categories:
            formatted_categories_yoy.append(f"<b>{category.strip()}</b>")
        elif category.startswith('  '):  # Indent subcategories
            formatted_categories_yoy.append(f"&nbsp;&nbsp;&nbsp;{category.strip()}")
        else:
            formatted_categories_yoy.append(category)

    # Create the YoY table (hidden by default)
    fig.add_trace(go.Table(
        name='YoY',
        columnwidth=[200] + [80 for _ in range(len(table_header_dates_yoy))],
        header=dict(
            values=["Category"] + table_header_dates_yoy,
            fill_color='black',
            font=dict(color='white', size=8),
            align='center',
            height=30
        ),
        cells=dict(
            values=[formatted_categories_yoy] + [table_df_yoy[col] for col in table_df_yoy.columns],
            fill=dict(color=color_fill_yoy),
            font=dict(color='white', size=8),
            align=['left'] + ['center'] * len(table_header_dates_yoy),
            height=20
        ),
        visible=False
    ))

    # Add frames for each window
    frames = []
    for i in range(min_windows):
        window_mom = windows_mom[i].transpose().round(1)
        window_yoy = windows_yoy[i].transpose().round(1)
        header_dates_mom = window_mom.columns.tolist()
        header_dates_yoy = window_yoy.columns.tolist()

        # Generate colors for MoM
        colors_mom_frame = [get_row_heatmap_colors(row) for _, row in window_mom.iterrows()]
        flat_colors_mom_frame = list(map(list, zip(*colors_mom_frame)))

        # Prepare fill colors for MoM
        color_fill_mom_frame = [['black'] * len(header_dates_mom)] + flat_colors_mom_frame

        # Format categories for MoM
        formatted_categories_mom_frame = []
        for category in window_mom.index:
            if category in bold_categories:
                formatted_categories_mom_frame.append(f"<b>{category.strip()}</b>")
            elif category.startswith('  '):
                formatted_categories_mom_frame.append(f"&nbsp;&nbsp;&nbsp;{category.strip()}")
            else:
                formatted_categories_mom_frame.append(category)

        # Generate MoM table frame
        frame_mom = go.Table(
            columnwidth=[200] + [80 for _ in range(len(header_dates_mom))],
            header=dict(
                values=["Category"] + header_dates_mom,
                fill_color='black',
                font=dict(color='white', size=8),
                align='center',
                height=20
            ),
            cells=dict(
                values=[formatted_categories_mom_frame] + [window_mom[col] for col in window_mom.columns],
                fill=dict(color=color_fill_mom_frame),
                font=dict(color='white', size=8),
                align=['left'] + ['center'] * len(header_dates_mom),
                height=20
            ),
            visible=True
        )

        # Generate colors for YoY
        colors_yoy_frame = [get_row_heatmap_colors(row) for _, row in window_yoy.iterrows()]
        flat_colors_yoy_frame = list(map(list, zip(*colors_yoy_frame)))

        # Prepare fill colors for YoY
        color_fill_yoy_frame = [['black'] * len(header_dates_yoy)] + flat_colors_yoy_frame

        # Format categories for YoY
        formatted_categories_yoy_frame = []
        for category in window_yoy.index:
            if category in bold_categories:
                formatted_categories_yoy_frame.append(f"<b>{category.strip()}</b>")
            elif category.startswith('  '):
                formatted_categories_yoy_frame.append(f"&nbsp;&nbsp;&nbsp;{category.strip()}")
            else:
                formatted_categories_yoy_frame.append(category)

        # Generate YoY table frame
        frame_yoy = go.Table(
            columnwidth=[200] + [80 for _ in range(len(header_dates_yoy))],
            header=dict(
                values=["Category"] + header_dates_yoy,
                fill_color='black',
                font=dict(color='white', size=8),
                align='center',
                height=20
            ),
            cells=dict(
                values=[formatted_categories_yoy_frame] + [window_yoy[col] for col in window_yoy.columns],
                fill=dict(color=color_fill_yoy_frame),
                font=dict(color='white', size=8),
                align=['left'] + ['center'] * len(header_dates_yoy),
                height=20
            ),
            visible=False
        )

        # Create the frame
        frame = go.Frame(
            data=[frame_mom, frame_yoy],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Define slider steps
    slider_steps = []
    for i, label in enumerate(window_labels):
        step = dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=label  # Label represents the start date of the window
        )
        slider_steps.append(step)

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Start Date: "},
        pad={"t": 50},
        steps=slider_steps,
        len=0.9,
        x=0.05,
        y=0
    )]

    fig.update_layout(
        sliders=sliders
    )

    # Add buttons for MoM and YoY tabs
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'label': 'MoM',
                    'method': 'update',
                    'args': [{'visible': [True, False]}, {'title': 'CPI MoM % Change'}]
                },
                {
                    'label': 'YoY',
                    'method': 'update',
                    'args': [{'visible': [False, True]}, {'title': 'CPI YoY % Change'}]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'x': 0.2,  # Position the buttons in the top left
            'xanchor': 'left',
            'y': 1.1,  # Position above the table
            'yanchor': 'top',
            'bgcolor': 'black',  # Set background to black
            'bordercolor': 'white',  # Set border to white
            'font': {'color': 'white'},  # White text on buttons
        }]
    )

    # Add source links as annotations below the table
    source_text = (
        "Sources: "
        "<a href='https://fred.stlouisfed.org/graph/?g=1uO7f' style='color:white'>PPI</a>, "
        "<a href='https://fred.stlouisfed.org/graph/?g=1uO8j' style='color:white'>Median Consumer Price Index</a>"
    )

    fig.add_annotation(
        text=source_text,
        showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.25,  # Positioned below the tables
        xanchor='center', yanchor='top',
        font=dict(size=8),
        align='center'
    )

    # Update layout for a black background and proper sizing
    fig.update_layout(
        width=1000,  # Adjust width to fit the tables
        height=700,  # Adjust height for better visibility
        template='plotly_dark',  # Use Plotly's dark template
        margin=dict(l=20, r=20, t=100, b=100),
    )

    return fig


# In[ ]:





# # Initialize Dash App and Deploy Web

# In[17]:


from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Initialize Dash app
app = Dash(__name__)

# Add custom CSS to set the entire page background to black and adjust container width to 1000px
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { 
                background-color: black !important;
                color: white;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                min-height: 100vh;
            }
            .container {
                width: 100%;
                max-width: 1000px; /* Set the max width to 1000px */
                padding: 0;
                box-sizing: border-box;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            .graph-container {
                display: block;
                width: 100%;
                text-align: center;
                padding: 0;
                box-sizing: border-box;
                margin: 0 auto;
            }
            .graph-container > div {
                display: inline-block;
                width: 100%;
                padding: 0;
                box-sizing: border-box;
                margin: 0 auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define a default period (e.g., 'YoY')
DEFAULT_PERIOD = 'YoY'

# App layout with adjusted container width and graph size
app.layout = html.Div(className='container', children=[
    html.H1("Economic Data Dashboard", style={'textAlign': 'center'}),

    # General data dropdown (CPI)
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': 'CPI', 'value': 'CPI'},
        ],
        value='CPI',
        style={
            'backgroundColor': '#000000',
            'color': 'white',  # Changed to white for better visibility on black background
            'width': '25%',
            'margin': '0 auto',
            'textAlign': 'center'
        }
    ),
    
    html.Div(id='cpi-graphs', children=[
        html.H2("Headline and Core CPI Growth Rate (YoY and 6mo Ann Graph 1)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-1', style={'width': '100%', 'max-width': '1000px'})
        ]),
        
        html.H2("CPI Components (Graph 2)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-2', style={'width': '100%', 'max-width': '1000px'})
        ]),
        
        html.H2("CPI Components (Graph 3)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-3', style={'width': '100%', 'max-width': '1000px'})
        ]),

        # Add summary table after Graph 3
        html.H2("CPI Growth Rate Table (MoM % Non-Weighted Table 1)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-summary-table-1', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("CPI Trend 6mo Annualized (Graph 4)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-4', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("Headline and Core PCE Growth Rate (YoY and 6mo Ann Graph 5)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-5', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("PCE Annualized Ex:Food and Energy (Graph 6)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-6', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("PPI Detailed Graph (Graph 7)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-7', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("PPI Components (Graph 8)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-8', style={'width': '100%', 'max-width': '1000px'})
        ]),
        
        html.H2("PPI (Graph 9)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-9', style={'width': '100%', 'max-width': '1000px'})
        ]),
        
        html.H2("Other Inflation Related Indicators (Table 1)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-table-1', style={'width': '100%', 'max-width': '1000px'})
        ]), 

        html.H2("CPI vs Surprise (Table 2)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-table-2', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("CPI Components vs Wage Growth (Graph 10)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-10', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("GDP Deflator (Graph 11)", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-11', style={'width': '100%', 'max-width': '1000px'})
        ]),

        html.H2("CPI Inflation Expectation", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            dcc.Graph(id='cpi-detailed-graph-12', style={'width': '100%', 'max-width': '1000px'})
        ]),
    
    ], style={'display': 'block'})
])

# Callbacks to manage visibility based on dropdown selection

# Callback for CPI Detailed Table 1
@app.callback(
    Output('cpi-detailed-table-1', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_table_1(selected_data):
    return generate_cpi_detailed_table_1() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Table 2
@app.callback(
    Output('cpi-detailed-table-2', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_table_2(selected_data):
    return generate_cpi_detailed_table_2() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 1
@app.callback(
    Output('cpi-detailed-graph-1', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_1(selected_data):
    return generate_cpi_detailed_graph_1() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 2 (Removed period-dropdown)
@app.callback(
    Output('cpi-detailed-graph-2', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_2(selected_data):
    return generate_cpi_detailed_graph_2() if selected_data == 'CPI' else go.Figure()

# Callback for CPI Detailed Graph 3 (Removed period-dropdown-graph-3)
@app.callback(
    Output('cpi-detailed-graph-3', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_3(selected_data):
     return generate_cpi_detailed_graph_3() if selected_data == 'CPI' else go.Figure()
      

# Callback for CPI Detailed Summary Table 1
@app.callback(
    Output('cpi-detailed-summary-table-1', 'figure'),
    [Input('data-dropdown', 'value')]  # Input to trigger update based on selected data
)
def update_cpi_detailed_summary_table_1(selected_data):
     return generate_cpi_detailed_summary_table_1() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 4
@app.callback(
    Output('cpi-detailed-graph-4', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_4(selected_data):
    return generate_cpi_detailed_graph_4() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 5
@app.callback(
    Output('cpi-detailed-graph-5', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_5(selected_data):
    return generate_cpi_detailed_graph_5() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 6
@app.callback(
    Output('cpi-detailed-graph-6', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_6(selected_data):
    return generate_cpi_detailed_graph_6() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 7
@app.callback(
    Output('cpi-detailed-graph-7', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_7(selected_data):
    return generate_cpi_detailed_graph_7() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 8
@app.callback(
    Output('cpi-detailed-graph-8', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_8(selected_data):
    return generate_cpi_detailed_graph_8() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 9
@app.callback(
    Output('cpi-detailed-graph-9', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_9(selected_data):
    return generate_cpi_detailed_graph_9() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 10
@app.callback(
    Output('cpi-detailed-graph-10', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_10(selected_data):
    return generate_cpi_detailed_graph_10() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 11
@app.callback(
    Output('cpi-detailed-graph-11', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_11(selected_data):
    return generate_cpi_detailed_graph_11() if selected_data == 'CPI' else go.Figure()


# Callback for CPI Detailed Graph 12
@app.callback(
    Output('cpi-detailed-graph-12', 'figure'),
    [Input('data-dropdown', 'value')]
)
def update_cpi_detailed_graph_12(selected_data):
    return generate_cpi_detailed_graph_12() if selected_data == 'CPI' else go.Figure()

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8042)


# In[ ]:





# In[36]:


import pandas as pd
import investpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Function to fetch economic calendar data for specific events
def fetch_event_data(event_name, from_date='01/01/2020', to_date='today'):
    # Fetch the global economic calendar data
    data = investpy.economic_calendar(from_date=from_date, to_date=to_date)
    # Filter for the specific event and the United States
    event_data = data[(data['event'].str.contains(event_name, case=False)) & (data['country'] == 'United States')]
    event_data['date'] = pd.to_datetime(event_data['date'])
    return event_data[['date', 'actual', 'forecast']]

# Fetch data for NFP, ADP, and Unemployment Rate
nfp_data = fetch_event_data("Nonfarm Payrolls")
adp_data = fetch_event_data("ADP Nonfarm Employment Change")
unemployment_data = fetch_event_data("Unemployment Rate")

# Generate the employment detailed graph
def generate_employment_detailed_graph(nfp_data, adp_data, unemployment_data):
    def process_event_data(event_data):
        # Ensure 'actual' and 'forecast' are numeric
        event_data['actual'] = pd.to_numeric(event_data['actual'], errors='coerce')
        event_data['forecast'] = pd.to_numeric(event_data['forecast'], errors='coerce')
        event_data.dropna(subset=['actual', 'forecast'], inplace=True)

        # Calculate the surprise
        event_data['surprise'] = np.where(
            event_data['forecast'] != 0,
            (event_data['actual'] - event_data['forecast']) / event_data['forecast'] * 100,
            np.nan
        )

        # Select the last 30 data points
        last_30_points = event_data.sort_values('date', ascending=False).head(30).sort_values('date')
        df_table = last_30_points[['date', 'actual', 'forecast', 'surprise']].copy()
        df_table.columns = ['Date', 'Actual', 'Forecast', 'Surprise']
        df_table['Date'] = df_table['Date'].dt.strftime('%y-%m')
        df_table.fillna(0, inplace=True)
        df_table_transposed = df_table.set_index('Date').transpose()
        return last_30_points, df_table_transposed

    # Process each data category
    nfp_points, nfp_table = process_event_data(nfp_data)
    adp_points, adp_table = process_event_data(adp_data)
    unemployment_points, unemployment_table = process_event_data(unemployment_data)

    # Create subplots
    fig = make_subplots(
        rows=8, cols=1,
        specs=[[{'type': 'xy'}], [{'type': 'xy'}], [{'type': 'table'}], [{'type': 'xy'}], [{'type': 'table'}],
               [{'type': 'xy'}], [{'type': 'table'}], [{'type': 'table'}]],
        subplot_titles=("Combined Graph", "NFP", "NFP Table", "ADP", "ADP Table", "Unemployment", "Unemployment Table"),
        vertical_spacing=0.05
    )

    # Combined Graph
    fig.add_trace(go.Scatter(x=nfp_points['date'], y=nfp_points['actual'], mode='lines', name='NFP Actual', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=adp_points['date'], y=adp_points['actual'], mode='lines', name='ADP Actual', line=dict(color='gray')), row=1, col=1)

    # NFP Graph with Forecast
    fig.add_trace(go.Scatter(x=nfp_points['date'], y=nfp_points['actual'], mode='lines', name='NFP Actual', line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=nfp_points['date'], y=nfp_points['forecast'], mode='lines', name='NFP Forecast', line=dict(color='cyan', dash='dot')), row=2, col=1)

    # NFP Table with heatmap
    if not nfp_table.empty:
        color_fill_nfp = create_color_fill(nfp_table)
        fig.add_trace(go.Table(
            header=dict(values=["Item"] + list(nfp_table.columns), fill_color='black', font=dict(color='white', size=8), align='center'),
            cells=dict(values=[nfp_table.index] + [nfp_table[col].round(2) for col in nfp_table.columns],
                       fill=dict(color=[['black'] * len(nfp_table.columns)] + color_fill_nfp), font=dict(color='white', size=8),
                       align='center', height=20)),
            row=3, col=1
        )

    # Repeat for ADP and Unemployment plots and tables, as shown in the previous examples

    # Update layout
    fig.update_layout(
        title="Employment Data Overview - NFP, ADP, Unemployment",
        height=1800,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=40),
        font=dict(size=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    
    return fig

# Generate the graph
fig = generate_employment_detailed_graph(nfp_data, adp_data, unemployment_data)
fig.show()


# In[ ]:





# In[ ]:




