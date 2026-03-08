"""
LLM Service
=============
Поднимает WebSocket сервер на эндпоинте /generate
Получает JSON формата: {"messages": [{"role": "system", "content": "..."}, ...]}
Возвращает стрим чанков ответов модели (OpenAI, Ollama, Stub) в формате JSON:
{"type": "chunk", "content": "..."}
В конце: {"type": "done"}
При ошибке: {"type": "error", "content": "текст"}
"""

import asyncio
import json
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("LLMService")

app = FastAPI(title="HSE LLM Service")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "stub").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
TRANSFORMERS_MODEL = os.environ.get("TRANSFORMERS_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = os.environ.get("DEVICE", "cpu")

# Глобальные переменные для кеширования модели (чтобы не грузить при каждом запросе)
_local_pipeline = None
_local_tokenizer = None


@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "LLM Service", "provider": LLM_PROVIDER}


@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("📡 Клиент подключён по WebSocket")
    try:
        data_str = await websocket.receive_text()
        data = json.loads(data_str)
        messages = data.get("messages", [])

        if LLM_PROVIDER == "openai":
            await _stream_openai(websocket, messages)
        elif LLM_PROVIDER == "ollama":
            await _stream_ollama(websocket, messages)
        elif LLM_PROVIDER == "local":
            await _stream_local_transformers(websocket, messages)
        else:
            await _stream_stub(websocket, messages)

        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("🔌 Клиент отключился")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass


async def _stream_openai(websocket: WebSocket, messages: list):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        await websocket.send_json({"type": "error", "content": "Библиотека openai не установлена"})
        return

    if not OPENAI_API_KEY:
        await websocket.send_json({"type": "error", "content": "ОШИБКА: Не задан OPENAI_API_KEY"})
        return

    client_ai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        stream = await client_ai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                await websocket.send_json({"type": "chunk", "content": content})
    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})


async def _stream_ollama(websocket: WebSocket, messages: list):
    try:
        import aiohttp
    except ImportError:
        await websocket.send_json({"type": "error", "content": "Библиотека aiohttp не установлена"})
        return

    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    err_txt = await resp.text()
                    await websocket.send_json({"type": "error", "content": f"Ollama error {resp.status}: {err_txt}"})
                    return
                
                async for line in resp.content:
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        await websocket.send_json({"type": "chunk", "content": chunk})
    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})


async def _stream_local_transformers(websocket: WebSocket, messages: list):
    global _local_pipeline, _local_tokenizer
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
        from threading import Thread
    except ImportError:
        await websocket.send_json({"type": "error", "content": "Библиотеки transformers/torch не установлены"})
        return

    # Инициализация модели при первом вызове
    if _local_pipeline is None:
        try:
            logger.info(f"⏳ Загружаю локальную модель: {TRANSFORMERS_MODEL} на {DEVICE}")
            _local_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL)
            # Упрощенная инициализация для TinyLlama и подобных
            _local_pipeline = pipeline(
                "text-generation",
                model=TRANSFORMERS_MODEL,
                tokenizer=_local_tokenizer,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device=0 if DEVICE == "cuda" else -1,
            )
            logger.info("✅ Локальная модель загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки локальной модели: {e}")
            await websocket.send_json({"type": "error", "content": f"Ошибка загрузки модели: {e}"})
            return

    try:
        # Подготовка промпта для TinyLlama (ChatML-подобный формат)
        # В реальном проекте стоит использовать tokenizer.apply_chat_template
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}</s>\n"
        prompt += "<|assistant|>\n"

        streamer = TextIteratorStreamer(_local_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        
        # Запускаем генерацию в отдельном потоке
        generation_kwargs = dict(
            input_ids=_local_tokenizer(prompt, return_tensors="pt").input_ids.to(_local_pipeline.device),
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_local_tokenizer.eos_token_id
        )
        
        thread = Thread(target=_local_pipeline.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Стримим токены из итератора
        for new_text in streamer:
            if new_text:
                await websocket.send_json({"type": "chunk", "content": new_text})
                await asyncio.sleep(0) # Даем шанс другим корутинам
                
    except Exception as e:
        logger.error(f"Local inference error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})


async def _stream_stub(websocket: WebSocket, messages: list):
    stub_text = "Я работаю в режиме заглушки (stub). Настройте LLM_PROVIDER в .env."
    for word in stub_text.split(" "):
        await websocket.send_json({"type": "chunk", "content": word + " "})
        await asyncio.sleep(0.1)

