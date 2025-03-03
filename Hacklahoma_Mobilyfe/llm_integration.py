import requests
from docx import Document
import PyPDF2
import io
from config import GEMINI_API_KEY
import streamlit as st

def extract_dates_events(syllabus_file):
    """
    Extract dates and events from a syllabus file as simple text.
    """
    try:
        # Get file content based on type
        content = ""
        file_type = syllabus_file.name.lower().split('.')[-1]

        if file_type == 'docx':
            doc = Document(io.BytesIO(syllabus_file.getvalue()))
            content = '\n'.join(paragraph.text for paragraph in doc.paragraphs)

        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(syllabus_file.getvalue()))
            content = '\n'.join(page.extract_text() for page in pdf_reader.pages)

        else:
            content = syllabus_file.getvalue().decode('utf-8', errors='ignore')

        # Call Gemini API
        prompt = f"""
        List all important dates and events from this syllabus.
        Include the date and what happens on that date.

        Syllabus content:
        {content}
        """

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse JSON response
        result = response.json()

        # Check if 'candidates' key exists and is not empty
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']

        return "No events found in the document."

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except ValueError as json_err:
        return f"JSON decode error: {json_err}"
    except KeyError as key_err:
        return f"Key error: {key_err}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def create_fake_events(events_text, start_date, end_date):
    """
    Create fake events with specific dates and times within a given date range.
    """
    try:
        # Call Gemini API to create fake events
        prompt = f"""
        Create fake events with specific dates and times within the date range {start_date} to {end_date} based on the following text.
        Ensure the events are evenly distributed throughout the semester.

        Events text:
        {events_text}
        """

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse JSON response
        result = response.json()

        # Check if 'candidates' key exists and is not empty
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']

        return "No events created."

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except ValueError as json_err:
        return f"JSON decode error: {json_err}"
    except KeyError as key_err:
        return f"Key error: {key_err}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Example usage
syllabus_file = ...  # Your syllabus file
events_text = extract_dates_events(syllabus_file)
start_date = "2025-01-15"
end_date = "2025-05-10"
fake_events = create_fake_events(events_text, start_date, end_date)
print(fake_events)
