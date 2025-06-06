# mygemini/stream_server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import Stream
from .gemini_handler import GeminiHandler
from pydantic import BaseModel
from typing import List, Optional

def create_gemini_stream(
    api_key: str,
    system_prompt: str,
    voice_name: str,
    expected_layout: str = "mono",
    output_sample_rate: int = 24000,
    output_frame_size: int = 480,
    input_sample_rate: int = 16000,
    ice_servers: Optional[List[dict]] = None,
    concurrency_limit: int = 50,
    time_limit: int = 900,
    cors_origins: List[str] = ["*"],
    cors_allow_credentials: bool = True,
    cors_allow_methods: List[str] = ["*"],
    cors_allow_headers: List[str] = ["*"],
) -> FastAPI:
    """
    Create and return a FastAPI app that hosts the Gemini streaming endpoint.

    The programmer sets the API key, system prompt, and voice name.
    The front-end only needs to send the webRTC id.
    
    Additional CORS settings can be provided via the `cors_*` parameters.
    """
    if ice_servers is None:
        ice_servers = [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"}
        ]

    # Instantiate the Gemini handler with the provided configuration.
    handler = GeminiHandler(
        api_key=api_key,
        system_prompt=system_prompt,
        voice_name=voice_name,
        expected_layout=expected_layout,
        output_sample_rate=output_sample_rate,
        output_frame_size=output_frame_size,
        input_sample_rate=input_sample_rate,
    )

    # Create the fastrtc Stream with the handler.
    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=handler,
        rtc_configuration={"iceServers": ice_servers},
        concurrency_limit=concurrency_limit,
        time_limit=time_limit,
    )

    # Define a pydantic model for input.
    class InputData(BaseModel):
        webrtc_id: str
        # voice_name and api_key are preconfigured by the programmer and not expected from the front-end.

    # Create the FastAPI app.
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=cors_allow_methods,
        allow_headers=cors_allow_headers,
    )

    # Mount the stream endpoints.
    stream.mount(app)

    # Define an endpoint that only accepts a webRTC id from the front-end.
    @app.post("/input_hook")
    async def input_hook(body: InputData):
        # Set the input using the provided webRTC id and the preconfigured API key and voice name.
        
        stream.set_input(body.webrtc_id, api_key, voice_name)
        return {"status": "ok"}

    return app
