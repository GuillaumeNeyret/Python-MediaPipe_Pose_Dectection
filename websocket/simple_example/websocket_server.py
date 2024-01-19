import asyncio
import websockets

async def envoyer_ping(websocket, path):
    while True:
        # Envoyer "Ping" à tous les clients connectés
        await asyncio.gather(
            *[client.send("Ping") for client in clients]
        )
        # Attendre 3 secondes avant d'envoyer le prochain "Ping"
        await asyncio.sleep(3)

start_server = websockets.serve(envoyer_ping, "localhost", 8765)

clients = set()

async def gestionnaire(websocket, path):
    print("Setp0")
    # Ajouter le client à la liste
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        # Retirer le client lorsque la connexion est fermée
        clients.remove(websocket)

start_server = websockets.serve(gestionnaire, "localhost", 8765)
print("Setp1")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
