import asyncio
import websockets
import random


async def hello(websocket):
    # name = await websocket.recv()
    # print(f'Server Received : {name}')
    # greeting = f'Hello {name}'

    await websocket.send('TEST')
    # print(f'Server Sent : {greeting}')

    # message = 'TEST'
    # await websocket.send(message)

# def random_number():
#     number = random.randint(0, 100)
#     return number

# async def send_random_number(websocket):
#     num = random_number()
#     await websocket.send(num)
#     print('Nombre envoyé')

async def main():
    websockets.serve(hello,"localhost",3000)
        # await asyncio.Future() # run forever
    # async with websockets.serve(send_random_number,"localhost",3000):
    #     await asyncio.Future() # run forever

if __name__ == "__main__":
    asyncio.run(main())