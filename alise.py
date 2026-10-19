import os
import sys
import json
import queue
import asyncio
import re
import sounddevice as sd
import pyttsx3
import httpx
from vosk import Model, KaldiRecognizer
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from duckduckgo_search import DDGS

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class AlisePersonalizedEngine:
    def __init__(self, config_path="config.json", profile_path="commander_profile.json"):
        self.config_path = config_path
        self.profile_path = profile_path
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.name = self.config["assistant_name"]
        self.wake_word = self.config["wake_word"].lower()
        self.api_model = self.config["api_model"]
        self.api_url = self.config["api_url"]
        self.sys_prompt = self.config["system_instruction"]
        
        # --- SECURE CREDENTIAL CHECK MATRIX ---
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("\n[CRITICAL SECURITY EXCEPTION]: 'GEMINI_API_KEY' missing from environment variables!")
            print("Please export your token into the active terminal instance before executing compilation loops.")
            sys.exit(1)
            
        self.audio_queue = queue.Queue()
        self.sample_rate = self.config["audio"]["sample_rate"]
        
        # Load offline Vosk data registers
        self.vosk_model = Model("model")
        self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        
        self.tts_engine = pyttsx3.init()
        self.chat_history = []
        self.websocket_client = None

    def _load_commander_profile(self) -> str:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'r') as f:
                return f"\n\nCURRENT ACTIVE COMMANDER PROFILE CONFIGS:\n{json.dumps(json.load(f), indent=2)}"
        return ""

    def voice_output(self, text: str):
        clean_speech = re.sub(r'```.*?```', '[Processing workspace matrix dataset]', text, flags=re.DOTALL)
        self.tts_engine.say(clean_speech)
        self.tts_engine.runAndWait()

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    async def mic_listening_daemon(self):
        stream = sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.config["audio"]["block_size"], 
                                  dtype='int16', channels=1, callback=self.audio_callback)
        with stream:
            while True:
                await asyncio.sleep(0.01)
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(audio_data):
                        res = json.loads(self.recognizer.Result())
                        raw_text = res.get("text", "")
                        if raw_text and self.wake_word in raw_text:
                            command = raw_text.replace(self.wake_word, "").strip()
                            if command:
                                await self.process_personalized_loop(command)

    async def _query_api(self, conversation):
        """Asynchronously dispatches formatted message contents to the Google Gemini endpoint."""
        gemini_contents = []
        for msg in conversation:
            if msg["role"] == "system":
                continue
            role_map = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({
                "role": role_map,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": gemini_contents,
            "systemInstruction": {
                "parts": [{"text": self.sys_prompt}]
            },
            "generationConfig": {
                "temperature": 0.6
            }
        }
        
        target_url = f"{self.api_url}/{self.api_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(target_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    return "Sir, my extraction matrix stumbled while parsing the logic output."
            return f"Gemini Gateway Error: Received Status Code {response.status_code} from routing network."

    async def execute_web_search(self, search_query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(search_query, max_results=3)]
                if results:
                    return "\n".join([f"Source: {r['title']} - Details: {r['body']}" for r in results])
        except Exception as e:
            print(f"Web harvest failure: {e}")
        return "The live web data harvest yielded no active parameters, Sir."

    async def process_personalized_loop(self, command: str):
        if self.websocket_client:
            await self.websocket_client.send_json({"event": "user_input", "text": command})
        self.chat_history.append({"role": "user", "content": command})
        if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]
        
        dynamic_profile = self._load_commander_profile()
        conversation = [{"role": "system", "content": self.sys_prompt + dynamic_profile}] + self.chat_history

        try:
            answer = await self._query_api(conversation)
            
            if "NEED_SEARCH:" in answer:
                parsed_query = answer.split("NEED_SEARCH:")[-1].strip()
                if self.websocket_client:
                    await self.websocket_client.send_json({"event": "alise_output", "text": f"Information gap discovered. Launching web scrapers for '{parsed_query}' real quick, Sir..."})
                
                search_payload = await self.execute_web_search(parsed_query)
                learning_context = f"LIVE INTERNET RESEARCH RESULT MATRIX:\n{search_payload}\n\nSynthesize this data and answer the Commander. Maintain your characteristic wit."
                self.chat_history.append({"role": "system", "content": learning_context})
                answer = await self._query_api([{"role": "system", "content": self.sys_prompt + dynamic_profile}] + self.chat_history[-6:])

            self.chat_history.append({"role": "assistant", "content": answer})
            if self.websocket_client:
                await self.websocket_client.send_json({"event": "alise_output", "text": answer})
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.voice_output, answer)
        except Exception as e:
            print(f"Assistant Loop Error: {e}")

engine = AlisePersonalizedEngine()

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine.websocket_client = websocket
    try:
        while True: await websocket.receive_text()
    except: engine.websocket_client = None

@app.on_event("startup")
async def app_startup():
    asyncio.create_task(engine.mic_listening_daemon())
