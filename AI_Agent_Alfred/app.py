import os
from dotenv import load_dotenv
load_dotenv()

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import pytz
import yaml
import gradio as gr
from typing import Any
from smolagents.tools import Tool

@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """A tool that concatenates a string and an integer
    Args:
        arg1: the first argument (string)
        arg2: the second argument (integer)
    """
    return f"Combined result: {arg1}{arg2}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

final_answer = FinalAnswerTool()
duckduckgo_search = DuckDuckGoSearchTool()
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

hf_token = os.getenv("HF_TOKEN")
print("Hugging Face Token:", hf_token)

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
    token=hf_token
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer, my_custom_tool, get_current_time_in_timezone, duckduckgo_search, image_generation_tool],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="Alfred",
    description="A versatile AI assistant capable of performing various tasks.",
    prompt_templates=prompt_templates
)

def predict(message):
    return agent.run(message)

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="AI Agent Alfred",
    description="Ask me anything!"
)

if __name__ == "__main__":
    iface.launch()
