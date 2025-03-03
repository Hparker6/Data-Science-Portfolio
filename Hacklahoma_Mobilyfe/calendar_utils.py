import streamlit as st

def list_calendars(service):
    calendar_list = service.calendarList().list().execute()
    calendars = calendar_list.get('items', [])
    for calendar in calendars:
        print(calendar['summary'])

def add_event_to_calendar(event, service):
    print(f"Event type: {type(event)}")  # This will show if event is a string or dict
    print(f"Event content: {event}")  # This will show the actual event content

    if isinstance(event, dict):  # Ensure event is a dictionary
        event_body = {
            'summary': event['summary'],
            'start': {'dateTime': event['start'], 'timeZone': 'America/Chicago'},
            'end': {'dateTime': event['end'], 'timeZone': 'America/Chicago'},
        }
        service.events().insert(calendarId='primary', body=event_body).execute()
        st.write(f"Added event: {event['summary']}")
    else:
        st.write("Error: event is not a dictionary!")
