# Automobile Sales Dashboard

This project creates an interactive dashboard to visualize automobile sales data over time, specifically focusing on yearly statistics and recession period data. The dashboard allows users to select different report types and years to view various visualizations, including line graphs, bar charts, and pie charts.

## Requirements

Before running the application, ensure the following Python packages are installed:

```bash
pip install dash
pip install plotly
pip install wget
```

These libraries are required to build the dashboard and visualize the data.

## Usage

1) **Install Dependencies**: Run the following commands in the terminal or in a Jupyter notebook cell to install the necessary packages:

```bash
%pip install dash
%pip install plotly
%pip install wget
```

2) **Import Libraries**: The script imports essential packages like `dash`, `plotly`, `pandas`, `requests`, and `numpy` for building the app and data manipulation.

3) **Run the Dashboard**: To start the dashboard, execute the following command in your terminal or Jupyter cell:

```bash
python app.py
```

This will start a local server, and the dashboard will be accessible in your browser at `http://127.0.0.1:8090`.

## Features

### Yearly Statistics: Displays:
- Line graph showing automobile sales over time.
- Bar chart for average sales by vehicle type.
- Pie chart for total sales share by vehicle type.
- Bar chart for advertising expenditures by vehicle type.

### Recession Period Statistics: Displays:
- Line graph showing automobile sales fluctuations during recession periods.
- Bar chart for average number of vehicles sold by vehicle type during recessions.
- Pie chart for advertising expenditure share by vehicle type during recessions.
- Bar chart showing the effect of unemployment rate on vehicle type and sales.

## Data Source

The data used for the dashboard is loaded from an external URL containing historical automobile sales data. It includes the following columns:

- `Year`
- `Month`
- `Vehicle_Type`
- `Automobile_Sales`
- `Advertising_Expenditure`
- `Recession`
- `unemployment_rate`

## Layout & Callback

The app layout is built using Dash's HTML and DCC components. Users can select a report type (Yearly or Recession Period Statistics) and a year to view visualizations. The app updates dynamically based on the selected options.

## How to Contribute

Feel free to fork this project, contribute improvements, or submit bug reports. Contributions are welcome to enhance the functionality of this dashboard, such as adding more statistics or improving the user interface.

## License

This project is licensed under the MIT License.
