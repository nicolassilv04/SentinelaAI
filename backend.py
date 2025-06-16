# backend.py (versão corrigida para garantir a integridade do CSV e robustez da previsão)
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import sys
import paho.mqtt.client as mqtt
import threading
import time
import yaml

# Módulos do projeto e de terceiros
import api_client
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

class DecisionTreePipeline:
    """Gerencia o treinamento e a predição do modelo de Árvore de Decisão."""
    def __init__(self, config: dict):
        self.config = config
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)
        self.encoder = LabelEncoder()
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> bool:
        """Treina o modelo com os dados históricos."""
        features_cols = self.config['feature_columns']
        target_col = self.config['target_column']
        
        df_train = df[features_cols + [target_col]].dropna()

        if len(df_train) < 20:
            logger.warning("IA - Árvore de Decisão: Dados insuficientes para o treinamento.")
            self.is_trained = False
            return False

        X = df_train[features_cols]
        y_encoded = self.encoder.fit_transform(df_train[target_col])
        
        self.model.fit(X, y_encoded)
        self.is_trained = True
        logger.info("IA - Árvore de Decisão: Modelo treinado com sucesso.")
        return True

    def predict(self, features: pd.DataFrame) -> Optional[str]:
        """Faz uma predição para um novo conjunto de dados."""
        if not self.is_trained:
            return "IA não treinada"
        
        prediction_encoded = self.model.predict(features)
        return self.encoder.inverse_transform(prediction_encoded)[0]

class Forecaster:
    """Gerencia o treinamento e a previsão de séries temporais."""
    def __init__(self, config: dict):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Treina um modelo para cada coluna alvo, com fallback para modelo não-sazonal."""
        target_cols = self.config['target_columns']
        trained_models = 0
        for col in target_cols:
            series = df[col].dropna()
            if len(series) > 10:
                try:
                    # --- INÍCIO DA CORREÇÃO ---
                    # Tenta primeiro o modelo completo com sazonalidade
                    model = ExponentialSmoothing(
                        series, 
                        trend='add', 
                        seasonal='add', 
                        seasonal_periods=12
                    ).fit()
                    self.models[col] = model
                    trained_models += 1
                    logger.info(f"IA - Previsão: Modelo SAZONAL treinado para '{col}'.")
                except ValueError:
                    # Se falhar (ex: por falta de dados), tenta um modelo mais simples sem sazonalidade
                    logger.warning(f"IA - Previsão: Modelo sazonal falhou para '{col}'. Tentando modelo não-sazonal mais simples.")
                    try:
                        model = ExponentialSmoothing(
                            series, 
                            trend='add', 
                            seasonal=None # Remove a sazonalidade
                        ).fit()
                        self.models[col] = model
                        trained_models += 1
                        logger.info(f"IA - Previsão: Modelo NÃO-SAZONAL treinado para '{col}'.")
                    except Exception as e_simple:
                        logger.error(f"IA - Previsão: Falha ao treinar até mesmo o modelo simples para '{col}'. Erro: {e_simple}")
                except Exception as e:
                    logger.error(f"IA - Previsão: Erro inesperado ao treinar modelo para '{col}'. Erro: {e}")
                    # --- FIM DA CORREÇÃO ---
        
        if trained_models > 0:
            self.is_trained = True
            logger.info(f"IA - Previsão: {trained_models} modelo(s) de série temporal treinado(s).")
        else:
            logger.warning("IA - Previsão: Nenhum modelo de previsão foi treinado.")


    def forecast(self) -> Optional[pd.DataFrame]:
        """Gera previsões para o horizonte definido."""
        if not self.is_trained:
            return None
        
        horizon = self.config['prediction_horizon_hours']
        forecast_data = {}
        # Adiciona um timestamp inicial para o índice da previsão
        last_known_timestamp = self.models[next(iter(self.models))].series.index[-1]
        forecast_index = pd.date_range(start=last_known_timestamp, periods=horizon + 1, freq='H')[1:]

        for col, model in self.models.items():
            forecast_values = model.forecast(horizon)
            # Garante que os valores tenham o índice correto
            forecast_data[col] = pd.Series(forecast_values, index=forecast_index)

        return pd.DataFrame(forecast_data)

class SentinelaVerde:
    """Classe principal que orquestra todo o fluxo de trabalho do backend."""
    def __init__(self, config: dict):
        self.config = config
        self.lock = threading.Lock()
        self.decision_tree = DecisionTreePipeline(self.config['models']['decision_tree'])
        self.forecaster = Forecaster(self.config['models']['forecasting'])
        self.df_data: Optional[pd.DataFrame] = None
        self.latest_classification: Optional[str] = "Aguardando..."
        self.future_forecast: Optional[pd.DataFrame] = None
        self.last_timestamp: Optional[datetime] = None
        self.page_update_callback: Optional[Callable[[], None]] = None
        # O agendador da API agora é iniciado pelo main_app para garantir que o loop de eventos Flet esteja rodando
        # self.start_api_scheduler() # REMOVIDO DAQUI

    def start_api_scheduler(self):
        """Inicia um agendador em background para buscar dados da API."""
        def run_scheduler():
            interval = self.config['api']['interval_seconds']
            while True:
                self.fetch_api_data_and_merge()
                time.sleep(interval)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="APIScheduler")
        scheduler_thread.start()


    def fetch_api_data_and_merge(self):
        """Busca dados da API e os salva."""
        try:
            api_conf = self.config['api']
            api_data = api_client.fetch_air_quality_data(api_conf['city'], api_conf['token'])
            if api_data:
                self.save_data_to_csv(api_data)
        except Exception as e:
            logger.error(f"Erro no agendador da API: {e}")

    def process_mqtt_message(self, topic: str, payload: str):
        try:
            parts = payload.split(',')
            if len(parts) != 3: return
            sensor_data = {
                'Temperatura_C': float(parts[0]),
                'Umidade_Relativa_percent': float(parts[1]),
                'Concentracao_Geral_PPM': float(parts[2]),
            }
            self.save_data_to_csv(sensor_data)
        except (ValueError, IndexError) as e:
            logger.error(f"Erro ao processar mensagem MQTT: {e}. Payload: '{payload}'")

    def save_data_to_csv(self, data_dict: Dict[str, Any]):
        """Salva um novo dicionário de dados no CSV, garantindo a ordem das colunas."""
        with self.lock:
            filepath = Path(self.config['files']['unified_csv'])
            
            df_historico = pd.read_csv(filepath) if filepath.exists() and filepath.stat().st_size > 0 else pd.DataFrame()

            data_dict['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df_new_row = pd.DataFrame([data_dict])

            df_final = pd.concat([df_historico, df_new_row], ignore_index=True)
            
            canonical_columns = ['Timestamp'] + self.config['models']['decision_tree']['feature_columns']
            df_final['Timestamp'] = pd.to_datetime(df_final['Timestamp'])
            
            # Remove linhas duplicadas com base no timestamp, mantendo a última ocorrência
            df_final.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
            df_final.sort_values(by='Timestamp', inplace=True)
            
            df_to_save = df_final.reindex(columns=canonical_columns)
            
            df_to_save.to_csv(filepath, index=False)
            
            logger.info(f"Dados salvos em {filepath}. Nova linha: {data_dict}")
            self.run_analysis()


    def load_data(self) -> bool:
        with self.lock:
            filepath = Path(self.config['files']['unified_csv'])
            if not filepath.exists() or filepath.stat().st_size < 10: return False
            
            # Define a frequência dos dados como horária ('H')
            self.df_data = pd.read_csv(filepath, parse_dates=['Timestamp'], index_col='Timestamp')
            if self.df_data.empty: return False
            
            self.df_data = self.df_data.asfreq('H') # Força uma frequência horária

            all_cols = self.config['models']['decision_tree']['feature_columns']
            for col in all_cols:
                if col not in self.df_data.columns: self.df_data[col] = np.nan
            
            self.df_data[all_cols] = self.df_data[all_cols].interpolate(method='linear').ffill().bfill()
            
            if self.df_data.empty: return False
            self.last_timestamp = self.df_data.index[-1].to_pydatetime()
            return True

    def run_analysis(self):
        if not self.load_data():
            logger.warning("Análise abortada: dados insuficientes ou arquivo vazio.")
            return

        df_copy = self.df_data.copy()
        limit = self.config['air_quality_limits']['Concentracao_Geral_PPM']
        df_copy['Qualidade_Ar_Calculada'] = np.where(df_copy['Concentracao_Geral_PPM'] > limit, 'Ruim', 'Bom')

        self.decision_tree.train(df_copy)
        self.forecaster.train(df_copy)

        latest_features = df_copy[self.config['models']['decision_tree']['feature_columns']].iloc[-1:]
        self.latest_classification = self.decision_tree.predict(latest_features)
        self.future_forecast = self.forecaster.forecast()
        
        logger.info(f"Análise concluída. Qualidade do ar atual: {self.latest_classification}")
        if self.page_update_callback: 
            self.page_update_callback()

    def get_latest_data_summary(self) -> Dict[str, Any]:
        if self.df_data is None or self.df_data.empty: return {}
        latest = self.df_data.iloc[-1]
        return {
            'timestamp': self.last_timestamp.strftime('%d/%m/%Y %H:%M') if self.last_timestamp else 'N/A',
            'Concentracao_Geral_PPM': latest.get('Concentracao_Geral_PPM'),
            'PM2.5_ug_m3': latest.get('PM2.5_ug_m3'), 'PM10_ug_m3': latest.get('PM10_ug_m3'),
            'Temperatura_C': latest.get('Temperatura_C'), 'Umidade_Relativa_percent': latest.get('Umidade_Relativa_percent'),
            'qualidade_ar': self.latest_classification, 'previsoes': self.future_forecast
        }

class MQTTClient:
    def __init__(self, config: dict, sentinela: SentinelaVerde):
        mqtt_conf = config['mqtt']
        self.sentinela = sentinela
        self.topic = mqtt_conf['topic']
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker = mqtt_conf['broker_address']
        self.port = mqtt_conf['port']

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.topic)
            logger.info(f"Conectado ao MQTT e assinando o tópico: '{self.topic}'")
        else:
            logger.error(f"Falha ao conectar ao MQTT, código: {rc}")

    def on_message(self, client, userdata, msg):
        self.sentinela.process_mqtt_message(msg.topic, msg.payload.decode())

    def start(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Não foi possível iniciar o cliente MQTT: {e}")
