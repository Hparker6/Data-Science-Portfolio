# Mobilyfe Calendar

Welcome to the Mobilyfe Calendar project! This application is designed to help you manage your time and health effectively. It features a calendar for scheduling events and a health dashboard for tracking your daily health metrics.

## Features

- **Calendar**: View and manage your events with different calendar modes (daygrid, timegrid, timeline, etc.).
- **Health Dashboard**: Track your heart rate, steps, sleep duration, and meditation time.
- **Data Visualization**: Visualize your weekly health data with bar charts.
- **User-Friendly Interface**: Easy navigation and input through a sidebar and interactive elements.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/mobilyfe-calendar.git
    cd mobilyfe-calendar
    ```

2. **Install the required packages**:
    ```bash
    pip install streamlit pandas matplotlib seaborn streamlit-calendar
    ```

3. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Usage

### Calendar

- Navigate to the "Calendar" page using the sidebar.
- Select the desired calendar mode from the dropdown menu.
- View and manage your events. Events can be customized with titles, colors, start and end dates, and resource IDs.

### Health Dashboard

- Navigate to the "Health Dashboard" page using the sidebar.
- Enter your daily health metrics (heart rate, steps, sleep duration, and meditation time).
- Submit your data to store it in the session state.
- View today's data and weekly averages.
- Visualize your weekly health data with a bar chart.

## Customization

You can customize the calendar options and events by modifying the `calendar_options` and `events` variables in the `app.py` file. Similarly, you can adjust the health dashboard input fields and data processing as needed.

## Acknowledgements

Thank you Hacklahoma! - Gaurav, Maya, Simon, and Houston

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
