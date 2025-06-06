# 🎙️ Gemini Live Voice Only Python

**Gemini Live Voice Only** is a real-time streaming library that integrates the Gemini API with FastAPI and WebRTC using `fastrtc`. This package enables developers to build applications that process audio streams in real time—ideal for tasks such as voice-based ID matching. 🔊

## ✨ Features

- 🚀 **Real-time Audio Streaming:** Seamlessly stream audio data via FastAPI and WebRTC.
- 🔍 **Gemini API Integration:** Leverage the Gemini API's live voice processing capabilities.
- ⚙️ **Customizable Configuration:** Set your API key, system prompt, and voice name at initialization.
- 🌍 **Flexible CORS Setup:** Configure CORS settings to suit your deployment needs.
- 🛠️ **Easy-to-Use API:** Quickly create and run your streaming server with minimal setup.

## 📥 Installation

Install the package from PyPI:

```bash
pip install gemini_live_voice_only
```

Or install it in editable mode from the source repository:

```bash
git clone <repository_url>
cd gemini_live_voice_only
pip install -e .
```

## 🚀 Usage

Here's a quick example to get you started:

```python
from gemini_live_voice_only import create_gemini_stream
import uvicorn

# 🎛️ Configuration parameters
API_KEY = "YOUR_API_KEY"
SYSTEM_PROMPT = (
    "You are a specialized ID matching tool. Compare input IDs with the reference ID 'RBTL24CB067' "
    "and output only the matching percentage."
)
VOICE_NAME = "Puck"

# 🎤 Create the FastAPI app with customizable CORS settings
app = create_gemini_stream(
    api_key=API_KEY,
    system_prompt=SYSTEM_PROMPT,
    voice_name=VOICE_NAME,
    cors_origins=["https://yourdomain.com"],  # specify allowed origins
    cors_allow_credentials=True,
    cors_allow_methods=["*"],
    cors_allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, reload=True)
```

## ⚙️ Configuration Options

When using the `create_gemini_stream` function, you can configure:

- 🔑 **API Key, System Prompt, and Voice Name:** Required for setting up the Gemini API connection.
- 🌐 **ICE Servers:** Customize the ICE server list for WebRTC connections if needed.
- 🔄 **CORS Settings:** Configure `cors_origins`, `cors_allow_credentials`, `cors_allow_methods`, and `cors_allow_headers` for cross-origin resource sharing.
- 🎶 **Audio Stream Parameters:** Adjust `expected_layout`, `output_sample_rate`, `output_frame_size`, and `input_sample_rate` based on your audio processing needs.
- ⚡ **Concurrency & Time Limits:** Modify `concurrency_limit` and `time_limit` to suit your application's requirements.

## 🤝 Contributing

Contributions are welcome! 🎉 If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request. 🚀
