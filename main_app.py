# main_app.py (versão corrigida que utiliza o config.yaml)
# Ponto de entrada principal da aplicação.

import backend
import frontend
import flet as ft
import threading
import logging
import yaml
import sys
from pathlib import Path

def load_config(config_path='config.yaml') -> dict:
    """
    Carrega o arquivo de configuração YAML.
    Encerra a aplicação se o arquivo não for encontrado.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"ERRO CRÍTICO: Arquivo de configuração '{config_path}' não encontrado.")
        print("Por favor, crie o arquivo 'config.yaml' e execute novamente.")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao ler o arquivo de configuração '{config_path}': {e}")
        sys.exit(1)

def setup_logging(config: dict):
    """
    Configura o sistema de logging com base no arquivo de configuração.
    """
    log_file = config.get('files', {}).get('log_file', 'sentinela_verde.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - (%(threadName)-10s) - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main(page: ft.Page):
    """
    Função principal que constrói e orquestra a aplicação.
    """
    # 1. Carrega as configurações do arquivo .yaml
    config = load_config()

    # 2. Configura o sistema de logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando a aplicação Sentinela Verde.")
    page.title = "Sentinela Verde"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # 3. Cria a instância ÚNICA e central do backend, passando APENAS a configuração.
    # A linha abaixo foi CORRIGIDA.
    sentinela = backend.SentinelaVerde(config)
    logger.info("Instância do SentinelaVerde criada.")

    # 4. Inicia o cliente MQTT em uma thread de segundo plano
    def mqtt_thread_worker():
        logger.info("Thread do cliente MQTT iniciada.")
        client = backend.MQTTClient(config, sentinela)
        client.start()

    mqtt_thread = threading.Thread(target=mqtt_thread_worker, name="MQTTThread", daemon=True)
    mqtt_thread.start()

    # 5. Chama a função principal do frontend para construir a UI.
    # O frontend.main é quem vai configurar o callback, não a criação do backend.
    frontend.main(page, sentinela)


if __name__ == "__main__":
    # Instale a dependência PyYAML se ainda não tiver: pip install PyYAML
    ft.app(target=main)
