from google.oauth2 import service_account
from googleapiclient.discovery import build

def authenticate_google(credentials_file):
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    service = build('calendar', 'v3', credentials=credentials)
    return credentials, service
