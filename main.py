from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import queue
import threading

from agent import run_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()

    data = await websocket.receive_json()
    query = data["query"]

    step_queue = queue.Queue()
    DONE = object()  # sentinel to signal "agent finished"

    def on_step(step):
        step_queue.put(step)

    def run_in_thread():
        run_agent(query, on_step=on_step)
        step_queue.put(DONE)

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    # poll the queue and forward steps to the browser as they arrive
    while True:
        try:
            step = step_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.1)  # nothing yet, wait a bit and check again
            continue

        if step is DONE:
            break

        await websocket.send_json(step)

    await websocket.close()