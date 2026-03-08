# LLM Service

Обособленный микросервис для генерации текста языковой моделью.
Реализует WebSocket-сервер для принятия промпта от `rag_service` и стриминга сгенерированного ответа обратно.

## Настройка `.env`

```env
# Выбор провайдера (stub | openai | ollama)
LLM_PROVIDER=stub

# Для OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Для Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Эндпоинты

- `GET /ping` — проверка статуса
- `WS /generate` — WebSocket эндпоинт, принимает JSON с `"messages": [...]` и отправляет события:
  - `{"type": "chunk", "content": "тест"}`
  - `{"type": "done"}`
  - `{"type": "error", "content": "..."}`

## Запуск

```bash
pip install -r requirements.txt
python -m uvicorn main:app --port 8002
```
