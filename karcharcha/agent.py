import io
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import ToolContext, FunctionTool
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from google.adk.agents import Agent
import numpy as np
from pydantic import BaseModel, Field
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any, Optional
#from google.adk.tools import google_search
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
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults
import mimetypes
import pathlib
from pathlib import Path
from . import constants
from . import tools
import tempfile
import wave
import pathlib
from pathlib import Path
import contextlib
import base64

# Instantiate the LangChain tool
tavily_tool_instance = TavilySearchResults(
    max_results=2,
    search_depth="basic",
    #return_direct=True
    
)
# Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)

# Custom tools
translate_and_speak_tool = FunctionTool(func = tools.translate_and_speak)
translate_to_english_tool = FunctionTool(func = tools.translate_to_english)

  
# --- 1. Define the Callback Function ---
def modify_output_after_agent(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Returns new Content to *replace* the agent's original output.
    
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()
    
    print(f"\n[Callback] Exiting agent: {agent_name} (Inv: {invocation_id})")
    #print(f"[Callback] Current State: {current_state}")

    parts = []
    audio_file = current_state.get("audio_file")
    answer = current_state.get("translated_answer")
    if audio_file:
        
        parts.append( types.Part.from_bytes(
            data=Path(audio_file).read_bytes(), mime_type='audio/wav',))

    if answer:
        parts.append(types.Part(text=answer))
    else:
        parts.append(types.Part(text="No answer found") )

    print("Returning modified content")
    return types.Content(role='model',parts=parts )       
    
# --- 2. Define Sub-Agents for Each Pipeline Stage ---

retreive_answer_agent = LlmAgent(
    name="get_answer",
    model=  constants.GEMINI_MODEL,
    instruction="""
     First always pass prompt to translate_to_english_tool.
     If a file has been provided then Query the file with the output of translate_to_english_tool
      and generate a  brief answer.
     Only if file is not provided then pass the output of translate_to_english_tool to the
      adk_tavily_tool and get a  brief answer.
    
    Output only  the retreived answer.
    """,
    description="Retreives answer for given query ",
   
    tools=[translate_to_english_tool, adk_tavily_tool],
    output_key="retreived_answer" ,
    
)
    
translate_speak_agent = LlmAgent(
    name="translate_speak",
    model=  constants.GEMINI_MODEL,
    instruction=
    """
     Pass the answer in the session state under the key "retreived_answer" to the translate_and_speak_tool.   
    """,
    description="Translates and speaks in users language",
    tools=[translate_and_speak_tool],
    after_agent_callback=modify_output_after_agent,
      
)

# --- 3. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.

karcharcha = SequentialAgent(
    name="karcharcha",
    description=(
        "Agent to transcribe audio to text and perform content retreival and answer in the language used by user"
    ),
    sub_agents=[ retreive_answer_agent, translate_speak_agent],
        
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=constants.APP_NAME, 
                                         user_id=constants.USER_ID, session_id=constants.SESSION_ID)
runner = Runner(agent=karcharcha, app_name=constants.APP_NAME, session_service=session_service)

root_agent = karcharcha

def blob_to_wav(blob_data, output_filename, channels=1, sample_width=2, frame_rate=8000):
    """
    Convert binary blob data to a WAV file.
    
    Parameters:
    -----------
    blob_data : bytes
        The binary audio data
    output_filename : str
        Path to save the WAV file
    channels : int, optional
        Number of audio channels (1 for mono, 2 for stereo)
    sample_width : int, optional
        Sample width in bytes (1, 2, or 4)
    frame_rate : int, optional
        Sampling frequency in Hz (e.g., 44100, 48000)
    """
    # Create an in-memory buffer for the audio data
    buffer = io.BytesIO(blob_data)
    
    # Create a new WAV file
    with wave.open(output_filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(buffer.getvalue())
    
    #print(f"WAV file successfully created at {output_filename}")

@contextlib.contextmanager
def wave_file(filename, channels=1, rate= 8000, #24000
               sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

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
        
    events = runner.run(user_id=constants.USER_ID, session_id=constants.SESSION_ID, new_message=content)
    text_answer=""
    audio_answer=""
    for event in events:
        if event.is_final_response() & (event.author == "translate_speak"):
            #print(f"len: {len(event.content.parts)}")
            for item in event.content.parts:
               
                if item.text:
                    #print(f"text: {item.text}")
                    text_answer = item.text

                if item.inline_data:
                    #print(f"inline_data: {item.inline_data.mime_type}")
                    #create wav file 
                    temp = tempfile.NamedTemporaryFile(suffix='.wav',dir="temp", delete=False)
                    out_wav_filename=temp.name
                    blob_to_wav(item.inline_data.data, out_wav_filename)
                    audio_answer = out_wav_filename
                
    return audio_answer, text_answer

#    

#     Please follow these steps in the <Steps> section and the constraints mentioned in the <Constraints> section
#     <Steps>
#     1. Pass prompt to translate_to_english_tool.
    
#     2. If a file has been provided then Query the file with the output of translate_to_english_tool
#       and generate a short answer and skip step 3 and go to step 4.
    
#     3. If file is not provided then pass the output of translate_to_english_tool to the
#       adk_tavily_tool and get a short answer.
    
#     4. Output only  the retreived answer.

#     </Steps>

#    <Constraints>
#     Do not mention your thoughts in the final response. Your final response should be less 
#     than 500 characters.
#     </Constraints>       
