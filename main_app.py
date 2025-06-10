# main_app.py
# Ponto de entrada principal da aplicação.
# Orquestra o backend, o cliente MQTT e a interface gráfica Flet.

import backend
import frontend
import flet as ft
import threading
import time
import logging

# --- Configurações ---
BROKER_ADDRESS = "test.mosquitto.org"
BROKER_PORT = 1883
# Tópicos que o backend irá assinar. Devem ser os mesmos que o ESP32 publica.
TOPICS_TO_SUBSCRIBE = ["sentinela/temperatura", "sentinela/umidade", "sentinela/gas"]

# --- Instâncias Globais ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Callback que será passado para o backend. Quando chamado, ele força a atualização da UI.
# É uma forma de comunicação entre a thread do backend e a thread da UI.
page_ref = ft.Ref[ft.Page]()

def update_page_from_thread():
    if page_ref.current and page_ref.current.session:
        # Acessa a página e chama o método de atualização
        page_ref.current.update()

# --- Orquestração ---
def main(page: ft.Page):
    # Armazena a referência da página para que outras threads possam atualizá-la
    page_ref.current = page

    # Cria a instância do Sentinela, passando a função de callback da UI
    sentinela = backend.SentinelaVerde(page_update_callback=update_page_from_thread)

    # Inicia o cliente MQTT em uma thread de segundo plano
    def mqtt_thread_worker():
        client = backend.MQTTClient(BROKER_ADDRESS, BROKER_PORT, TOPICS_TO_SUBSCRIBE, sentinela)
        client.start()
        logger.info("Thread do cliente MQTT iniciada.")
        # Mantém a thread viva
        while True:
            time.sleep(1)
            
    mqtt_thread = threading.Thread(target=mqtt_thread_worker, daemon=True)
    mqtt_thread.start()

    # Chama a função principal do frontend para construir a UI
    frontend.main(page, sentinela)


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("Iniciando Sentinela Verde com MQTT...")
    # Inicia a aplicação Flet. A função 'main' acima será chamada.
    ft.app(target=main)
