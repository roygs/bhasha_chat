from shutil import copy
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from google.adk.agents import Agent
from google.adk.tools import ToolContext, FunctionTool
from google.adk.planners import BuiltInPlanner
from google.genai import types
from typing import Any, List, Optional
import json
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any
#from google.adk.tools import google_search
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults

from pydantic import BaseModel, Field
import pathlib
from pathlib import Path
import contextlib
import wave
import base64
import mimetypes
import io
import requests

import mimetypes
import os
import threading

from dotenv import load_dotenv

from fastapi import File
from . import constants
from . import prompt

import tempfile
from sarvamai import SarvamAI
from sarvamai.play import play, save
 
# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

load_dotenv()

# --- Constants ---

# TEMP_DIR = Path(constants.TEMP_FOLDER) 
# TEMP_DIR.mkdir(exist_ok=True)
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#TEMP_DIR = os.path.join(BASE_DIR, "temp")

api_key = os.environ['SARVAM_API_KEY']

# #input schema for the translate_and_speak_tool
  
class InfoAnswer(BaseModel):
    language: str = Field(description="The language in which the user asked the query")
    retreived_answer: str = Field(description="Answer obtained by querying the document")
    
def parse_adk_output_to_dictionary(events: list[Event]) -> dict[str, Any]:
    """
    Parse ADK event output into a structured dictionary format,
    with the predicted trajectory dumped as a JSON string.

    """

    final_response = ""
    predicted_trajectory_list = []

    for event in events:
        # Ensure content and parts exist before accessing them
        if not event.content or not event.content.parts:
            continue

        # Iterate through ALL parts in the event's content
        for part in event.content.parts:
            if part.function_call:
                tool_info = {
                    "tool_name": part.function_call.name,
                    "tool_input": dict(part.function_call.args),
                }
                # Ensure we don't add duplicates if the same call appears somehow
                if tool_info not in predicted_trajectory_list:
                    predicted_trajectory_list.append(tool_info)

            # The final text response is usually in the last event from the model
            if event.content.role == "model" and part.text:
                # Overwrite response; the last text response found is likely the final one
                final_response = part.text.strip()
                

    # Dump the collected trajectory list into a JSON string
    final_output = {
        "response": str(final_response),
        "predicted_trajectory": json.dumps(predicted_trajectory_list),
    }

    return final_output

def format_output_as_markdown(output: dict) -> str:
    """Convert the output dictionary to a formatted markdown string."""
    markdown = "### AI Response\n"
    markdown += f"{output['response']}\n\n"

    if output["predicted_trajectory"]:
        output["predicted_trajectory"] = json.loads(output["predicted_trajectory"])
        markdown += "### Function Calls\n"
        for call in output["predicted_trajectory"]:
            markdown += f"- **Function**: `{call['tool_name']}`\n"
            markdown += "  - **Arguments**:\n"
            for key, value in call["tool_input"].items():
                markdown += f"    - `{key}`: `{value}`\n"

    return markdown


def get_lang_code(language:str):
    if len(language) == 2:
        return language
    
    return constants.ALL_LANG_CODES.get(language.lower())

#helper function
def get_translation(text, language):
    """
    Translate the text to the given language
    Returns: translated text
    """
    client = SarvamAI(api_subscription_key=api_key)    

    target_lang_code = get_lang_code(language)
    if target_lang_code not in constants.INDIAN_LANG_CODES:
        translated_text="User has asked question in a language which is not supported."
        target_lang_code = "en-IN"
    else:
        target_lang_code = target_lang_code + "-IN"
        if target_lang_code != "en-IN":
            # if target is not english then translate

            response = client.text.translate(
                    input=text,
                    source_language_code="en-IN",
                    target_language_code=target_lang_code,
                    speaker_gender="Male",
                    mode="classic-colloquial",
                    model="mayura:v1",
                    enable_preprocessing=False,
                )
            translated_text = response.translated_text
        else:
            translated_text = text

    return translated_text, target_lang_code

def get_audio_answer():
    ans="dummy"
    return ans

def speak(text, lang_code):
    """
    Converts text to speech
    Returns: wav file path
    """

    client = SarvamAI(api_subscription_key=api_key)    

    response = client.text_to_speech.convert( inputs=[text],
        target_language_code=lang_code, speaker="anushka", enable_preprocessing=True,
    )
    temp = tempfile.NamedTemporaryFile(suffix='.wav',dir="temp", delete=False,delete_on_close=False)
    filename=temp.name
    save(response, filename)
    return filename

 
# Tool function
def translate_and_speak(a_dict: InfoAnswer, tool_context:ToolContext) :
    """
        Translate the text and convert to speech.
        Returns: 
            translated_answer
    """
    
    translated_answer, lang_code = get_translation(a_dict.get("retreived_answer"), a_dict.get("language"))
    
   
    audio_answer = speak(translated_answer, lang_code)
    
    return {"translated_answer":translated_answer, "audio_answer":audio_answer}

# Instantiate the LangChain tool
tavily_tool_instance = TavilySearchResults(
    max_results=1,
    search_depth="basic",
    #return_direct=True
    
)

# Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)

translate_and_speak_tool = FunctionTool(func = translate_and_speak)
get_audio_answer_tool = FunctionTool(func = get_audio_answer)

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]

# generate_content_config = types.GenerateContentConfig(
#    #safety_settings=safety_settings,
#    #temperature=0.28,
#    #max_output_tokens=500,
#    #top_p=0.95,
#    thinking_config=types.ThinkingConfig(thinking_budget=0)
  
# )

def after_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    """Inspects/modifies the tool result after execution."""
    agent_name = tool_context.agent_name
    tool_name = tool.name
    
    print(f"[Callback] After tool call for tool '{tool_name}' in agent '{agent_name}'")
    if tool_name == "translate_and_speak":
        
        original_result_value = tool_response
 
        print(f"[Callback] tool response : {original_result_value}")   
        tool_context.state['audio_filename'] = original_result_value.get('audio_answer')
        print(f"Saving state audio_filename: {tool_context.state['audio_filename']}")
        # Return the modified dictionary
        #modified_response = original_result_value.get('translated_answer')
        #return modified_response
        return None

        
    if tool_name == "get_audio_answer":
        original_result_value = tool_response
        print(f"[Callback] tool response : {original_result_value}")   
        modified_response = tool_context.state['audio_filename']
        print(f"[Callback] Modified tool response : {modified_response}")
        return modified_response



bhasha_chat_agent = Agent(
    name="bhasha_chat_agent",
    model=constants.GEMINI_MODEL,
    description=(
        "Agent to transcribe audio to text and perform content retreival and answer in the language used by user"
    ),
    #generate_content_config=generate_content_config,
    
    tools=[adk_tavily_tool, translate_and_speak_tool,get_audio_answer_tool],
    instruction=prompt.ROOT_PROMPT,
    after_tool_callback=after_tool_modifier,
    #generate_content_config=generate_content_config
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=1024,
        )
    ),
    
     
    
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=constants.APP_NAME, 
                                         user_id=constants.USER_ID, session_id=constants.SESSION_ID)
runner = Runner(agent=bhasha_chat_agent, app_name=constants.APP_NAME, session_service=session_service)

root_agent = bhasha_chat_agent

# Agent Interaction
async def call_agent(data_file, query_file, user_prompt=""):
 
    final_response="error"
    parts=[]
    
    if data_file:

        mime_type, encoding = mimetypes.guess_type(data_file)
        # add the data file 
        parts=[ 
            types.Part.from_bytes(
            data=Path(data_file).read_bytes(),mime_type=mime_type,)]
            
    if query_file:
        # Case 1: user has recorded his query.
   
        # ignore user_prompt if user has recorded query
        prompt=""
        parts.append( types.Part.from_bytes(
            data=Path(query_file).read_bytes(), mime_type='audio/wav',))
       
    else:
        # Case 2: query.wav not present, i.e. user has typed his query.
        if user_prompt == "":
            # Case 3: Neither audio nor text query present
            final_response = "Please type or ask a question."
            return final_response
        else:
            prompt=user_prompt
        
    parts.append(types.Part(text=prompt))
    content = types.Content(role='user',parts=parts )
        
    #events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    events = list(
        runner.run(user_id=constants.USER_ID, session_id=constants.SESSION_ID, new_message=content)
    )
    #response = parse_adk_output_to_dictionary(events)
    #print( format_output_as_markdown(response))
    text_answer=""
    audio_answer=""
    for event in events:
        
        if event.is_final_response():
            
            if event.content.parts[0]:
                print(f"Event.content.parts[0] {event.content.parts[0]}")
                final_response = event.content.parts[0].text.split("`")
                
                text_answer = final_response[0]
                audio_answer = final_response[1]
                
                
    return audio_answer, text_answer



# FILES_FOLDER=".\\files"
# df = os.path.join(FILES_FOLDER, "Principal-Sample-Life-Insurance-Policy.pdf")
# qf = os.path.join(FILES_FOLDER,"q_en.wav")

# r = call_agent(df, "", "summarize this document")
# #r = call_agent(df, "", "sfsgss")
# #r = call_agent(df, "", "how are you today")
# print (f"Agent Response: \n {r}")

# query="हिताधिकारी को क्या मिलेगा"

#df = os.path.join(FILES_FOLDER, "Principal-Sample-Life-Insurance-Policy.pdf")
# qf = os.path.join(FILES_FOLDER,"q_hi.wav")

# r = call_agent(df, qf)
# #r = call_agent(df, "", "summarize this document")
# r = call_agent(df, "", "what will beneficiary get")
# # #r = call_agent("",  qf)
# # #r = call_agent(df, qf,"how are you") #"how are you"
# # #r = call_agent(df, "","tum kaise ho")
# # #r = call_agent(df, qf,"how are you")
# #r = call_agent(df, "", query)
# print (f"Agent Response: \n {r}")