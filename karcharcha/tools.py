import tempfile
from sarvamai import SarvamAI
from sarvamai.play import play, save
from google.adk.tools.base_tool import BaseTool
from google.adk.tools import ToolContext, FunctionTool
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

import logging
from pydub import AudioSegment
from . import constants
import pathlib
from pathlib import Path
import contextlib
import wave
import base64
import mimetypes
import io
import requests
import os
from dotenv import load_dotenv

@contextlib.contextmanager
def wave_file(filename, channels=1, rate= 8000, #24000
               sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

load_dotenv()



api_key = os.environ['SARVAM_API_KEY']
client = SarvamAI(api_subscription_key=api_key)    

def get_lang_code(language:str):
    if len(language) == 2:
        return language
    
    return constants.ALL_LANG_CODES.get(language.lower())


def get_translation(text, language):
    """
    Translate the text to the given language
    Returns: translated text
    """

    if language == "en-IN":
        return text
    #target_lang_code = get_lang_code(language)
    
    #if target_lang_code not in constants.INDIAN_LANG_CODES:
    #    translated_text="User has asked question in a language which is not supported."
        
    #else:
        #target_lang_code = target_lang_code + "-IN"
        #if target_lang_code != "en-IN":
            # if target is not english then translate
    target_lang_code = language
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
    
    return translated_text


def speak(text:str, lang_code):
    """
    Converts text to speech
    Returns: wav file path
    """

    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
    "inputs": [text],
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.25,
    "loudness": 2,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1",
    "target_language_code": lang_code
    }
    headers = {
        'api-subscription-key': api_key,
        "Content-Type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()
    print("Request successful")

    audio_data = response.json()['audios']
    delimiter = ""
    audio_string = delimiter.join(audio_data)
    

    temp = tempfile.NamedTemporaryFile(suffix='.wav',dir="temp", delete=False)
    out_wav_filename=temp.name
    with wave_file(out_wav_filename) as wav:
        audio_bytes = base64.b64decode(audio_string)
        wav.writeframes(audio_bytes)
       
    return out_wav_filename


# Tool function


def translate_to_english(prompt:str, tool_context: ToolContext):
    """
        Translate the text to english. Saves the source language in the state
        Returns:
            translated_query
    """
    #print(f"User prompt : {prompt}")
    response = client.text.translate(
            input=prompt,
            source_language_code="auto",
            target_language_code="en-IN",
            speaker_gender="Male",
            mode="classic-colloquial",
            model="mayura:v1",
            enable_preprocessing=False,
        )
    translated_text = response.translated_text
    lang_code = response.source_language_code
 
    #print(f"Translated query: {translated_text}")
    #print(f"source lang_code: {lang_code}")

    tool_context.state['lang_code'] = lang_code
    return {"translated_query": translated_text }
   


def translate_and_speak(answer:str, tool_context:ToolContext) :
    """
        Translate the text and convert to speech.
        Saves translated answer and name of audio file in the state.
        Returns: 
            translated_answer
    """
    print("translating and speaking")
    try :
        target_lang = tool_context.state.get('lang_code')

        #print(f"Language : {target_lang}")
        #print(f"Retrieved answer : {answer}")
        if target_lang == None:
            target_lang = "en-IN"
            translated_answer = answer
        else:
            translated_answer = get_translation(answer, target_lang)
        #print(f"Translated answer: {translated_answer}")
    
        audio_answer = speak(translated_answer, target_lang)
        
        tool_context.state['audio_file'] =  audio_answer
        tool_context.state['translated_answer'] = translated_answer

        return {"result":"success", "translated_answer":translated_answer}
    except Exception as e:
        print(f"Error translating and speaking: {e}")
        return {"result":"error", "message":e}

