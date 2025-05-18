from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uvicorn
import datetime # For unique filenames
import os       # For getting file extension
from contextlib import asynccontextmanager
#from bhasha_chat_agent.agent import call_agent
from karcharcha.agent import call_agent



# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Create uploads directory if it doesn't exist
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)  # Create temp directory if it doesn't exist

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # code to execute when app is loading
    print("loading app")
    yield
    # code to execute when app is shutting down
    print("closing")
    clear_files()
    
app = FastAPI(lifespan=app_lifespan)    

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
# Mount uploads directory to make files accessible via URL (for audio playback, etc.)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")




# Setup Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")






# --- In-memory "database" for chat messages (for demo purposes) ---
# Each message can have: type, sender, content, and optionally path/filename
chat_messages = []
files_uploaded = []
audio_files = []

# --- Helper to get file extension from content type ---
def get_extension_from_content_type(content_type: str) -> str:
    if not content_type:
        return ".bin" # Default binary extension
    # Common audio types often have parameters like 'audio/wav;codecs=opus'
    main_type = content_type.split(';')[0]
    return "." + main_type.split('/')[-1]

def get_filename(filepath: str):
    if not filepath:
        return ""
    return Path(filepath).name

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Serves the main chat page."""
    # Pass existing messages to the template
    # For audio messages, ensure 'path' is included if you want them to be playable on load
    # For this example, we'll just pass them as they are stored
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_messages})

@app.post("/send_message/")
async def handle_send_message(request: Request, message: str = Form(...)):
    """Handles text messages sent from the client."""
    #print(f"Received text message: {message}")
    if message:
        chat_messages.append({"type": "text", "sender": "User", "content": message})
    # In a real app, you'd broadcast this to other users (e.g., via WebSockets)
    
    data_file = files_uploaded[-1] if len(files_uploaded) != 0 else ""
    query_file = audio_files[-1] if len(audio_files) != 0 else ""
    audio_result, text_result = await call_agent(data_file, query_file, message)
    audio_files.clear()
    filename = get_filename(audio_result)
    #print(filename)
    relative_path = f"/temp/{filename}"
    #print(f"Call agent returned {result}")
    chat_messages.append({"type": "text", "sender": "Agent", "content": text_result})
    data = {"message": text_result, "status": "success","filename": filename, "path": relative_path}
    return JSONResponse(content=data)
    

@app.post("/upload_file/")
async def handle_upload_file(request: Request, file: UploadFile = File(...)):
    """Handles general file uploads."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize filename
    safe_filename_base = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in Path(file.filename).stem)
    safe_extension = "".join(c if c.isalnum() or c == '.' else '' for c in Path(file.filename).suffix)
    if not safe_extension and file.content_type: # Try to get extension from content type if not in filename
        ext_from_type = get_extension_from_content_type(file.content_type)
        if ext_from_type != ".bin": # Avoid generic .bin if possible
            safe_extension = ext_from_type

    filename = f"{timestamp}_{safe_filename_base}{safe_extension if safe_extension else '.dat'}"
    file_location = UPLOAD_DIR / filename

    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        print(f"File '{filename}' uploaded successfully to {file_location}")
        chat_messages.append({"type": "file", "sender": "User", "content": f"Uploaded: {filename}", "filename": filename})
        files_uploaded.append(file_location)
        return JSONResponse(content={"status": "File uploaded", "filename": filename})
    except Exception as e:
        print(f"Error uploading file: {e}")
        return JSONResponse(content={"status": "File upload failed", "error": str(e)}, status_code=500)
    finally:
        if file and hasattr(file, 'file') and not file.file.closed:
            file.file.close()

@app.post("/upload_audio/")
async def handle_upload_audio(request: Request, audio: UploadFile = File(...)):
    """Handles uploaded audio recordings."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_extension = get_extension_from_content_type(audio.content_type)
    if not file_extension or file_extension == ".bin": # Fallback for generic types
        # Try to infer from filename if provided by client (e.g., 'recording.wav')
        original_filename = audio.filename if audio.filename else "audio_recording"
        _, ext = os.path.splitext(original_filename)
        file_extension = ext if ext else ".wav" # Default to wav if no good info

    # Ensure extension starts with a dot
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    filename = f"audio_{timestamp}{file_extension}"
    file_location = UPLOAD_DIR / filename

    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(audio.file, file_object)
        print(f"Audio file '{filename}' uploaded successfully to {file_location}")
        # Store the relative path for client-side playback
        relative_path = f"/uploads/{filename}"
        chat_messages.append({"type": "audio", "sender": "User", "content": f"Recorded audio: {filename}", "path": relative_path, "filename": filename})
        audio_files.append(file_location)
        return JSONResponse(content={"status": "Audio uploaded", "filename": filename, "path": relative_path})
    except Exception as e:
        print(f"Error uploading audio: {e}")
        return JSONResponse(content={"status": "Audio upload failed", "error": str(e)}, status_code=500)
    finally:
        if audio and hasattr(audio, 'file') and not audio.file.closed:
            audio.file.close()

def clear_files( ):
    print("clearing")
    # Clear the contents of the upload directory
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")            




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")