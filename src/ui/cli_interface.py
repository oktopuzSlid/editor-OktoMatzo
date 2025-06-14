def main_cli():
    print("Bienvenido al sistema de edición visual inteligente.")
    print("Escribe un comando (ej: 'recorta al gato'):")
    while True:
        command = input("> ")
        if command in ("salir", "exit"):
            break
        print(f"Comando recibido: {command} (procesamiento pendiente)")
