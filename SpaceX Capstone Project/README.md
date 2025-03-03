# SpaceX Capstone Project

This project focuses on analyzing SpaceX launch data through web scraping, exploratory data analysis (EDA), interactive dashboards, and predictive modeling. It demonstrates a comprehensive approach to data science, from data collection to visualization and prediction.

## Project Overview

- **Web Scraping**: Collected SpaceX launch data from public websites
- **EDA**: Performed analysis using Python and SQL
- **Dashboards**: Created interactive visualizations with Dash and Plotly
- **Visualizations**: Utilized Seaborn and Folium for additional insights
- **Predictive Models**: Developed and tuned machine learning models to forecast launch outcomes

## Requirements

Ensure the following Python packages are installed:
```pip install dash plotly seaborn folium pandas numpy scikit-learn requests beautifulsoup4```

## Key Components

### Web Scraping
- Scraped data includes launch date, rocket type, launch site, mission details, and outcomes
- Data cleaned and processed for analysis

### Exploratory Data Analysis (EDA)
- Utilized Python (Pandas, Seaborn) and SQL
- Analyzed launch success rates, rocket types, and launch sites
- Explored relationships between launch variables

### Dashboards and Visualizations
- Interactive dashboards using Dash and Plotly
- Visualizations include line charts, bar charts, pie charts, and maps
- Additional visualizations with Seaborn and Folium

### Predictive Models
- Developed models to forecast launch outcomes
- Used algorithms like Logistic Regression, Random Forest, and XGBoost
- Performed model testing and hyperparameter tuning

## Usage Instructions

1. Install dependencies:
```pip install -r requirements.txt```

2. Run web scraping script:
```python scrape_data.py```

3. Launch the Dash app:
```python app.py```

Access the dashboard at `http://127.0.0.1:8090`

## Contributing
Contributions for enhancing functionality or improving the user interface are welcome. Please fork the project and submit pull requests.

## License
This project is licensed under the MIT License.

