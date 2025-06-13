# api_client.py (versão que usa token passado como argumento)
# -*- coding: utf-8 -*-

import requests
import pandas as pd
from datetime import datetime
import os
import logging
from pathlib import Path

# A configuração de logging agora será feita pelo backend.

def fetch_air_quality_data(cidade: str, token: str) -> dict:
    """
    Busca os dados de qualidade do ar (PM2.5, PM10) para uma cidade, usando o token fornecido.
    """
    logging.info(f"Buscando dados de PM2.5 e PM10 para a cidade: {cidade}...")
    url = f"https://api.waqi.info/feed/{cidade}/?token={token}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logging.error(f"API retornou um erro: {data.get('data', 'Erro desconhecido')}")
            return {}

        iaqi = data["data"].get("iaqi", {})
        dados_poluentes = {
            'PM2.5_ug_m3': iaqi.get('pm25', {}).get('v'),
            'PM10_ug_m3': iaqi.get('pm10', {}).get('v'),
        }

        dados_validos = {k: v for k, v in dados_poluentes.items() if v is not None}
        if dados_validos:
            logging.info(f"Dados da API externa recebidos com sucesso: {dados_validos}")
        else:
            logging.warning("A API não retornou dados de PM2.5 ou PM10 para a cidade.")
            
        return dados_validos

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de conexão ao tentar acessar a API: {e}")
        return {}
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado ao processar os dados da API: {e}")
        return {}

# A função de salvar no CSV foi movida para o backend para centralizar o controle.
# O bloco if __name__ == "__main__": foi removido, pois este script agora é apenas um módulo.
