# gerador_de_dados.py
# Este script simula o hardware (ESP32 + Sensores) enviando dados via MQTT.
# É útil para testar o backend e o frontend sem precisar do dispositivo físico.

import paho.mqtt.client as mqtt
import time
import random
import yaml
from pathlib import Path

def load_mqtt_config():
    """Carrega as configurações do MQTT do arquivo config.yaml."""
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("ERRO: O arquivo 'config.yaml' não foi encontrado. Execute este script no mesmo diretório do seu projeto.")
        exit()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['mqtt']

def connect_mqtt(config):
    """Cria e conecta o cliente MQTT."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = lambda c, u, f, rc: print("Gerador de Dados: Conectado ao Broker MQTT com sucesso!") if rc == 0 else print(f"Gerador de Dados: Falha ao conectar, código {rc}")
    
    try:
        client.connect(config['broker_address'], config['port'], 60)
        return client
    except Exception as e:
        print(f"ERRO: Não foi possível conectar ao broker MQTT em {config['broker_address']}:{config['port']}. Verifique sua conexão. Erro: {e}")
        return None

def generate_and_publish_data():
    """
    Gera dados simulados de sensores e os publica no tópico MQTT.
    """
    mqtt_config = load_mqtt_config()
    client = connect_mqtt(mqtt_config)
    
    if not client:
        return

    client.loop_start()
    
    print("--- Iniciando o Gerador de Dados para Teste ---")
    print(f"Enviando dados para o tópico: '{mqtt_config['topic']}'")
    print("Pressione CTRL+C para parar.")

    try:
        while True:
            # Gera valores realistas, mas aleatórios
            # Temperatura entre 20°C e 35°C
            temperatura = round(random.uniform(20.0, 35.0), 2)
            # Umidade entre 40% e 70%
            umidade = round(random.uniform(40.0, 70.0), 2)
            # Concentração de gás com picos ocasionais
            # A maioria dos valores será baixa (bom), com 10% de chance de ser alta (ruim)
            if random.random() < 0.1:
                 concentracao = round(random.uniform(350.0, 800.0), 2) # Qualidade ruim
            else:
                 concentracao = round(random.uniform(50.0, 250.0), 2) # Qualidade boa

            # Formata a mensagem no mesmo padrão do Arduino: "temp,umid,gas"
            payload = f"{temperatura},{umidade},{concentracao}"

            # Publica a mensagem
            result = client.publish(mqtt_config['topic'], payload)
            
            # Verifica se a publicação foi bem-sucedida
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"-> Dados enviados: temp={temperatura}°C, umid={umidade}%, gas_ppm={concentracao}")
            else:
                print(f"Falha ao enviar dados. Código de erro: {result.rc}")

            # Espera 10 segundos antes de enviar o próximo dado
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n--- Gerador de Dados encerrado ---")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    generate_and_publish_data()
