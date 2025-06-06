from gemini_live_voice_only import create_gemini_stream
import uvicorn

# Set the configuration parameters.
API_KEY = "AIzaSyChFAdb_g6eJLni537PJp1k5emo-f3LGRA"
SYSTEM_PROMPT = (
    "You are a helpful assistant and your name is omiii and created by omanand.co, Use given function declarations as tools to perform operations."
)
VOICE_NAME = "Puck"

# Create the FastAPI app.
app = create_gemini_stream(
    api_key=API_KEY,
    system_prompt=SYSTEM_PROMPT,
    voice_name=VOICE_NAME,

)

if __name__ == "__main__":
    uvicorn.run(app, port=7860)
