import streamlit as st
from auth import authenticate_google
from calendar_utils import list_calendars, add_event_to_calendar
from llm_integration import extract_dates_events

# Streamlit UI
st.title("Syllabus to Calendar App")
st.write("Upload your syllabus and let the app extract important dates and events.")
syllabus = st.file_uploader("Upload your syllabus", type=["pdf", "docx"])

# Google Calendar API setup
credentials, service = authenticate_google('token.json')

# Main logic
if syllabus:
    dates_events = extract_dates_events(syllabus)
    st.write(dates_events)
    
    # Check if the returned dates_events is a list of dictionaries
    if isinstance(dates_events, list):
        for event in dates_events:
            if isinstance(event, dict):  # Ensure each event is a dictionary
                add_event_to_calendar(event, service)
                st.write(f"Added event: {event['summary']}")
            else:
                st.write(f"Skipping invalid event: {event}")
    else:
        st.write("Error: No events found or invalid format!")

st.button("List Calendars", on_click=list_calendars, args=(service,))
