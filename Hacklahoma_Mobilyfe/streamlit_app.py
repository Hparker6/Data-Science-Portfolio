import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_calendar import calendar
from streamlit_folium import st_folium
import folium
import openrouteservice as ors
import requests
from typing import List, Dict
from datetime import datetime
import time
from quickstart import fetch_calendar_events
import os

from map import mapping_call

def configure_page():
    st.set_page_config(page_title="Mobilyfe Calendar", page_icon="📆")
    if "events" not in st.session_state:
        temp = [{"title" : event[0], "start" : str(event[1]).replace(" ", "T"), "end" : str(event[2]).replace(" ", "T")}
                                    for event in fetch_calendar_events()]

        st.session_state.events = temp
    if "health_data" not in st.session_state:
        st.session_state.health_data = []

def display_title():
    st.title("Mobilyfe")
    st.markdown("## Your Time, Your Mind, Your Lyfe")

def display_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Calendar", "Health Dashboard", "Location Finder"])

def display_calendar():
    st.markdown("### Calendar 📆")
    mode = st.selectbox(
        "Calendar Mode:",
        (
            "daygrid",
            "timegrid",
            "timeline",
            "resource-daygrid",
            "resource-timegrid",
            "resource-timeline",
            "list",
            "multimonth",
        ),
    )

    calendar_resources = [
        {"id": "a", "building": "Building A", "title": "Room A"},
        {"id": "b", "building": "Building A", "title": "Room B"},
    ]

    calendar_options = {
        "editable": "true",
        "navLinks": "true",
        "resources": calendar_resources,
        "selectable": "true",
    }

    if "resource" in mode:
        if mode == "resource-daygrid":
            calendar_options.update({
                "initialDate": "2023-07-01",
                "initialView": "resourceDayGridDay",
                "resourceGroupField": "building",
            })
        elif mode == "resource-timeline":
            calendar_options.update({
                "headerToolbar": {"left": "today prev,next", "center": "title", "right": "resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth"},
                "initialDate": "2023-07-01",
                "initialView": "resourceTimelineDay",
                "resourceGroupField": "building",
            })
        elif mode == "resource-timegrid":
            calendar_options.update({
                "initialDate": "2023-07-01",
                "initialView": "resourceTimeGridDay",
                "resourceGroupField": "building",
            })
    else:
        if mode == "daygrid":
            calendar_options.update({
                "headerToolbar": {"left": "today prev,next", "center": "title", "right": "dayGridDay,dayGridWeek,dayGridMonth"},
                "initialDate": "2023-07-01",
                "initialView": "dayGridMonth",
            })
        elif mode == "timegrid":
            calendar_options.update({"initialView": "timeGridWeek"})
        elif mode == "timeline":
            calendar_options.update({
                "headerToolbar": {"left": "today prev,next", "center": "title", "right": "timelineDay,timelineWeek,timelineMonth"},
                "initialDate": "2023-07-01",
                "initialView": "timelineMonth",
            })
        elif mode == "list":
            calendar_options.update({"initialDate": "2023-07-01", "initialView": "listMonth"})
        elif mode == "multimonth":
            calendar_options.update({"initialView": "multiMonthYear"})

    state = calendar(
        events=st.session_state.events,
        options=calendar_options,
        custom_css="""
        .fc-event-past { opacity: 0.8; }
        .fc-event-time { font-style: italic; }
        .fc-event-title { font-weight: 700; }
        .fc-toolbar-title { font-size: 2rem; }
        """,
        key=mode,
    )

    if state.get("eventsSet") is not None:
        st.session_state.events = state["eventsSet"]

    st.write(state)

def display_health_dashboard():
    st.markdown("### Health Dashboard 📊")
    bpm = st.number_input("Enter your heart rate (BPM):", min_value=0)
    steps = st.number_input("Enter your steps for the day:", min_value=0)
    sleep = st.number_input("Enter your sleep duration (hours):", min_value=0.0, step=0.1)
    meditation = st.number_input("Enter your meditation duration (minutes):", min_value=0)

    if st.button("Submit"):
        st.session_state.health_data.append({
            "date": pd.Timestamp.now().date(),
            "bpm": bpm,
            "steps": steps,
            "sleep": sleep,
            "meditation": meditation,
        })
        st.success("Data submitted successfully!")

    st.write("### Today's Data")
    st.write(f"Heart Rate: {bpm} BPM")
    st.write(f"Steps: {steps}")
    st.write(f"Sleep: {sleep} hours")
    st.write(f"Meditation: {meditation} minutes")

    if st.session_state.health_data:
        df = pd.DataFrame(st.session_state.health_data)
        df["date"] = pd.to_datetime(df["date"])
        last_week = df[df["date"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]

        if not last_week.empty:
            weekly_averages = last_week.mean(numeric_only=True)

            st.write("### Weekly Averages")
            st.write(f"Heart Rate: {weekly_averages['bpm']:.2f} BPM")
            st.write(f"Steps: {weekly_averages['steps']:.2f}")
            st.write(f"Sleep: {weekly_averages['sleep']:.2f} hours")
            st.write(f"Meditation: {weekly_averages['meditation']:.2f} minutes")

            last_week["date"] = last_week["date"].dt.strftime('%b-%d')

            st.write("### Weekly Data Chart")
            sns.set_palette("pastel")
            fig, ax = plt.subplots()
            last_week.set_index("date").plot(kind='bar', ax=ax)
            st.pyplot(fig)

def find_locations_old(lat: float, lon: float, radius: int, location_type: str, mobility_mode: str) -> List[Dict]:
    api_key = st.secrets.get("OPENROUTESERVICE_API_KEY", "")
    if not api_key:
        st.error("OpenRouteService API key not configured!")
        return []

    client = ors.Client(key=api_key)

    try:
        query = f"""
        [out:json];
        (
          node["amenity"="{location_type}"](around:{radius},{lat},{lon});
        );
        out body;
        """

        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for element in data['elements']:
            try:
                route = client.directions(
                    coordinates=[[lon, lat], [element['lon'], element['lat']]],
                    profile=mobility_mode,
                    format='geojson'
                )

                results.append({
                    'name': element.get('tags', {}).get('name', 'Unnamed'),
                    'lon': element['lon'],
                    'lat': element['lat'],
                    'distance': route['features'][0]['properties']['summary']['distance'],
                    'duration': route['features'][0]['properties']['summary']['duration']
                })
                time.sleep(0.1)

            except Exception as e:
                st.warning(f"Could not calculate route for {element.get('tags', {}).get('name', 'Unnamed')}: {str(e)}")
                continue

        return results

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching locations: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return []

def display_location_finder():
    st.markdown("### Location Finder 🗺️")

    lat = st.number_input("Enter latitude:", value=0.0, step=0.000001, format="%.6f")
    lon = st.number_input("Enter longitude:", value=0.0, step=0.000001, format="%.6f")
    radius = st.number_input("Enter radius (in meters):", value=1000, min_value=100, step=100)
    location_type = st.selectbox("Select location type:", ["restaurant", "park", "sports", "socialize"])
    mobility_mode = st.selectbox("Select mobility mode:", ["foot", "cycle", "car"])

    if st.button("Find Locations"):
        results = mapping_call(lat, lon, radius, location_type, mobility_mode)
        display_results(results, lat, lon)


def display_results(results: List[Dict], center_lat: float, center_lon: float):
    if not results:
        st.warning("No locations found in the specified area.")
        return

    st.write(f"Found {len(results)} locations:")
    
    # Convert results to DataFrame for better display
    df = pd.DataFrame(results)
    
    # Format distance and duration
    df['distance'] = df['distance'].round(0).astype(int)
    df['duration'] = (df['duration'] / 60).round(1)
    
    # Rename columns for better readability
    df = df.rename(columns={
        'name': 'Name',
        'distance': 'Distance (m)',
        'duration': 'Duration (min)',
        'lat': 'Latitude',
        'lon': 'Longitude'
    })
    
    # Reorder columns
    columns = ['Name', 'Distance (m)', 'Duration (min)', 'Latitude', 'Longitude']
    df = df[columns]
    
    # Display as a Streamlit table
    st.dataframe(
        df,
        column_config={
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Distance (m)": st.column_config.NumberColumn("Distance (m)", format="%d"),
            "Duration (min)": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
            "Latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
            "Longitude": st.column_config.NumberColumn("Longitude", format="%.6f")
        },
        hide_index=True
    )



# def display_results(results: List[Dict], center_lat: float, center_lon: float):
#     if not results:
#         st.warning("No locations found in the specified area.")
#         return

#     st.write(f"Found {len(results)} locations:")

#     try:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
#         folium.Marker(
#             [center_lat, center_lon],
#             popup="Your Location",
#             icon=folium.Icon(color='red')
#         ).add_to(m)

#         for result in results:
#             folium.Marker(
#                 [result['lat'], result['lon']],
#                 popup=f"""
#                     <b>{result['name']}</b><br>
#                     Distance: {result['distance']:.0f}m<br>
#                     Duration: {result['duration']/60:.1f}min
#                 """
#             ).add_to(m)

#         st_folium(m)

#         for result in results:
#             st.write(
#                 f"### {result['name']}\n"
#                 f"- Distance: {result['distance']:.0f}m\n"
#                 f"- Duration: {result['duration']/60:.1f} minutes"
#             )

    # except Exception as e:
    #     st.error(f"Error displaying map: {str(e)}")

def main():
    file = open("key.txt", "r")
    api_key = str(file.read())
    os.environ["OPRS_API_KEY"] = api_key
    
    configure_page()
    display_title()
    page = display_sidebar()
    if page == "Calendar":
        display_calendar()
    elif page == "Health Dashboard":
        display_health_dashboard()
    elif page == "Location Finder":
        display_location_finder()
    st.write("Thank you Hacklahoma! -Gaurav, Maya, Simon, and Houston")

if __name__ == "__main__":
    main()
