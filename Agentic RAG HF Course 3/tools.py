from langchain_community.tools import DuckDuckGoSearchRun 
from langchain.tools import Tool
import random
from huggingface_hub import list_models

def get_weather_info(location: str) -> str:
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

def get_hub_stats(author: str) -> str:
    try:
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

search_tool = DuckDuckGoSearchRun()
weather_info_tool = Tool(name="get_weather_info", func=get_weather_info, description="Dummy weather data by location.")
hub_stats_tool = Tool(name="get_hub_stats", func=get_hub_stats, description="Most downloaded HF model by author.")

tools = [search_tool, weather_info_tool, hub_stats_tool]
