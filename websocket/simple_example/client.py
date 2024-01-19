import asyncio
import websockets

async def hello():
    url = "ws://localhost:3000"
    async with websockets.connect(url) as websocket:
        name = input("what's your name ? ")
        await websocket.send(name)

        message = await websocket.recv() # wait for the serveur message
        print(f'Client received : {message}')

if __name__ == "__main__":
    asyncio.run(hello())