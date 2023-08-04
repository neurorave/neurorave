import threading
import asyncio
import json

import websockets

class Network(threading.Thread):
    def __init__(self, rx_queue, data, ip="", port=8001):
        super(Network, self).__init__()
        self.rx_queue = rx_queue
        self.data = data
        self.ip = ip
        self.port = port

    def run(self):
        asyncio.run(self.do_turn())

    async def handler(self, websocket):
        await websocket.send(json.dumps(self.data))
        async for message in websocket:
            self.rx_queue.put(json.loads(message))

    async def do_turn(self):
        async with websockets.serve(self.handler, self.ip, self.port):
            await asyncio.Future()  # run forever