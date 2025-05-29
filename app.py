# -*- coding: utf-8 -*-
"""
Sentinela Verde - Sistema Avançado de Monitoramento e Predição da Qualidade do Ar
Versão Otimizada com melhorias em robustez, configurabilidade e performance

Desenvolvido para análise de dados de sensores IoT com classificação em tempo real
e predição futura usando redes neurais LSTM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys


Sequential=None

# Configurar warnings e logging
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinela_verde.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Gerenciador de configurações do sistema"""
    
    DEFAULT_CONFIG = {
        'files': {
            'input_csv': 'meus_dados_arduino.csv',
            'output_classified_csv': 'dados_arduino_classificados.csv',
            'output_prediction_json': 'sentinela_dashboard_data_multialvo.json',
            'config_file': 'config.yaml'
        },
        'columns': {
            'sensors': [
                'Amonia_ppm',
                'Benzeno_ppm', 
                'Alcool_ppm',
                'Dioxido_Carbono_ppm',
                'Temperatura_C',
                'Umidade_Relativa_percent'
            ],
            'timestamp': 'Timestamp'
        },
        'air_quality_limits': {
            'Amonia_ppm': 0.1,
            'Benzeno_ppm': 0.002,
            'Alcool_ppm': 0.5,
            'Dioxido_Carbono_ppm': 1000
        },
        'sensor_ranges': {
            'Amonia_ppm': {'min': 0, 'max': 50},
            'Benzeno_ppm': {'min': 0, 'max': 10},
            'Alcool_ppm': {'min': 0, 'max': 100},
            'Dioxido_Carbono_ppm': {'min': 0, 'max': 5000},
            'Temperatura_C': {'min': -40, 'max': 85},
            'Umidade_Relativa_percent': {'min': 0, 'max': 100}
        },
        'lstm': {
            'enabled': True,
            'look_back': 48,
            'prediction_horizon': 24,
            'target_columns': ['Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'],
            'train_split': 0.8,
            'validation_split': 0.15,
            'epochs': 70,
            'batch_size': 32,
            'patience': 12,
            'min_data_points': 100
        }
    }
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self.DEFAULT_CONFIG['files']['config_file']
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Carrega configuração do arquivo YAML ou usa padrão"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                logger.info(f"Configuração carregada de {self.config_path}")
                return self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                logger.warning(f"Erro ao carregar configuração: {e}. Usando configuração padrão.")
        else:
            logger.info("Arquivo de configuração não encontrado. Criando configuração padrão.")
            self.save_default_config()
        
        return self.DEFAULT_CONFIG.copy()
    
    def save_default_config(self):
        """Salva configuração padrão em arquivo YAML"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuração padrão salva em {self.config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração padrão: {e}")
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Mescla configuração carregada com padrão"""
        merged = default.copy()
        for key, value in loaded.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

class DataValidator:
    """Validador de dados dos sensores"""
    
    def __init__(self, sensor_ranges: Dict[str, Dict[str, float]]):
        self.sensor_ranges = sensor_ranges
    
    def validate_sensor_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Valida dados dos sensores e remove outliers"""
        validation_stats = {}
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if column in self.sensor_ranges:
                initial_count = len(df_clean)
                sensor_range = self.sensor_ranges[column]
                
                # Remove valores fora do range físico
                mask = (df_clean[column] >= sensor_range['min']) & (df_clean[column] <= sensor_range['max'])
                df_clean = df_clean[mask]
                
                # Estatísticas de validação
                removed_count = initial_count - len(df_clean)
                validation_stats[column] = removed_count
                
                if removed_count > 0:
                    logger.warning(f"Removidos {removed_count} valores inválidos da coluna {column}")
        
        return df_clean, validation_stats
    
    def detect_anomalies(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Detecta anomalias usando método IQR"""
        df_clean = df.copy()
        
        for column in columns:
            if column in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[column]):
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    logger.info(f"Detectados {outliers_count} outliers em {column}")
                    # Marcar outliers mas não remover (apenas log)
                    df_clean[f'{column}_outlier'] = outliers_mask
        
        return df_clean

class AirQualityClassifier:
    """Classificador de qualidade do ar"""
    
    def __init__(self, limits: Dict[str, float]):
        self.limits = limits
        self.categories = {
            (0, 0.5): "Bom",
            (0.5, 1.0): "Regular", 
            (1.0, 1.5): "Ruim",
            (1.5, float('inf')): "Crítico"
        }
    
    def calculate_air_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula índice e categoria de qualidade do ar"""
        df_result = df.copy()
        pollutants = [col for col in self.limits.keys() if col in df_result.columns]
        
        if not pollutants:
            logger.error("Nenhum poluente encontrado para classificação")
            return df_result
        
        logger.info(f"Calculando qualidade do ar para: {pollutants}")
        
        # Calcular sub-índices
        sub_indices = pd.DataFrame(index=df_result.index)
        for pollutant in pollutants:
            df_result[pollutant] = pd.to_numeric(df_result[pollutant], errors='coerce')
            sub_indices[f'SubIndice_{pollutant}'] = df_result[pollutant] / self.limits[pollutant]
        
        # Índice máximo (pior caso)
        df_result['Max_SubIndice'] = sub_indices.max(axis=1)
        
        # Categorização
        df_result['Qualidade_Ar_Calculada'] = df_result['Max_SubIndice'].apply(self._categorize)
        df_result['Risco_Saude'] = df_result['Max_SubIndice'].apply(self._assess_health_risk)
        
        # Estatísticas
        self._log_quality_stats(df_result)
        
        return df_result
    
    def _categorize(self, max_sub_index: float) -> str:
        """Categoriza qualidade do ar baseado no sub-índice máximo"""
        if pd.isna(max_sub_index):
            return "Indeterminado"
        
        for (min_val, max_val), category in self.categories.items():
            if min_val <= max_sub_index < max_val:
                return category
        
        return "Crítico"
    
    def _assess_health_risk(self, max_sub_index: float) -> str:
        """Avalia risco à saúde"""
        if pd.isna(max_sub_index):
            return "Indeterminado"
        elif max_sub_index <= 0.5:
            return "Baixo"
        elif max_sub_index <= 1.0:
            return "Moderado"
        elif max_sub_index <= 1.5:
            return "Alto"
        else:
            return "Muito Alto"
    
    def _log_quality_stats(self, df: pd.DataFrame):
        """Log estatísticas de qualidade do ar"""
        if 'Qualidade_Ar_Calculada' in df.columns:
            stats = df['Qualidade_Ar_Calculada'].value_counts()
            logger.info("Distribuição da Qualidade do Ar:")
            for category, count in stats.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  {category}: {count} ({percentage:.1f}%)")

class LSTMPredictor:
    """Preditor LSTM para múltiplos alvos"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.target_columns = config['target_columns']
        self.is_trained = False
        
        # Verificar disponibilidade das bibliotecas
        self.ml_available = self._check_ml_libraries()
    
    def _check_ml_libraries(self) -> bool:
        """Verifica se bibliotecas de ML estão disponíveis"""
        try:
            global MinMaxScaler, Sequential, LSTM, Dense, Dropout, EarlyStopping, plt
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            import matplotlib.pyplot as plt
            
            logger.info("Bibliotecas de Machine Learning disponíveis")
            return True
        except ImportError as e:
            logger.warning(f"Bibliotecas ML não disponíveis: {e}")
            return False
    
    def prepare_data(self, df: pd.DataFrame, sensor_columns: List[str]) -> bool:
        """Prepara dados para treinamento LSTM"""
        if not self.ml_available:
            return False
        
        # Features válidas (numéricas)
        self.feature_columns = [
            col for col in sensor_columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # Verificar se targets estão nas features
        missing_targets = [col for col in self.target_columns if col not in self.feature_columns]
        if missing_targets:
            logger.error(f"Colunas alvo não encontradas nas features: {missing_targets}")
            return False
        
        # Preparar DataFrame
        self.df_lstm = df[self.feature_columns].copy()
        self.df_lstm.dropna(inplace=True)
        
        if len(self.df_lstm) < self.config['min_data_points']:
            logger.warning(f"Dados insuficientes para LSTM: {len(self.df_lstm)} < {self.config['min_data_points']}")
            return False
        
        logger.info(f"Dados preparados para LSTM: {len(self.df_lstm)} amostras, {len(self.feature_columns)} features")
        return True
    
    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências temporais para LSTM"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df_lstm)
        self.scaler = scaler
        
        look_back = self.config['look_back']
        horizon = self.config['prediction_horizon']
        
        X_sequences, y_sequences = [], []
        target_indices = [self.df_lstm.columns.get_loc(col) for col in self.target_columns]
        
        for i in range(len(scaled_data) - look_back - horizon + 1):
            X_sequences.append(scaled_data[i:(i + look_back), :])
            y_slice = scaled_data[(i + look_back):(i + look_back + horizon), :]
            y_sequences.append(y_slice[:, target_indices])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Constrói modelo LSTM otimizado"""
        model = Sequential([
            LSTM(units=80, return_sequences=True, input_shape=input_shape, activation='tanh'),
            Dropout(0.3),
            LSTM(units=60, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            LSTM(units=40, return_sequences=False, activation='tanh'),
            Dropout(0.2),
            Dense(units=len(self.target_columns) * self.config['prediction_horizon']),
            # Reshape para (horizon, n_targets)
        ])
        
        model.compile(
            optimizer='adam',
            loss='huber',  # Mais robusto a outliers que MSE
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, sensor_columns: List[str]) -> bool:
        """Treina modelo LSTM"""
        if not self.ml_available or not self.prepare_data(df, sensor_columns):
            return False
        
        logger.info("Iniciando treinamento LSTM...")
        
        try:
            X, y = self.create_sequences()
            
            if X.shape[0] == 0:
                logger.error("Não foi possível criar sequências para treinamento")
                return False
            
            # Reshape y para formato apropriado
            y = y.reshape(y.shape[0], -1)
            
            # Divisão treino/teste
            split_idx = int(X.shape[0] * self.config['train_split'])
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Divisão: Treino={X_train.shape[0]}, Teste={X_test.shape[0]}")
            
            # Construir e treinar modelo
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                shuffle=False,
                verbose=1
            )
            
            # Avaliar modelo
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            logger.info(f"Loss - Treino: {train_loss[0]:.4f}, Teste: {test_loss[0]:.4f}")
            logger.info(f"MAE - Treino: {train_loss[1]:.4f}, Teste: {test_loss[1]:.4f}")
            
            # Salvar gráfico de treinamento
            self._plot_training_history(history)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Erro durante treinamento LSTM: {e}")
            return False
    
    def predict_future(self, last_readings: pd.DataFrame) -> Optional[np.ndarray]:
        """Faz predições futuras"""
        if not self.is_trained or self.scaler is None:
            logger.error("Modelo não treinado ou scaler não disponível")
            return None
        
        try:
            # Preparar entrada
            input_data = last_readings[self.feature_columns].iloc[-self.config['look_back']:].values
            input_scaled = self.scaler.transform(input_data)
            input_reshaped = np.expand_dims(input_scaled, axis=0)
            
            # Predição
            prediction_scaled = self.model.predict(input_reshaped, verbose=0)
            
            # Reshape para (horizon, n_targets)
            prediction_reshaped = prediction_scaled.reshape(
                self.config['prediction_horizon'], 
                len(self.target_columns)
            )
            
            # Inverter escalonamento
            target_indices = [self.df_lstm.columns.get_loc(col) for col in self.target_columns]
            dummy_prediction = np.zeros((prediction_reshaped.shape[0], len(self.feature_columns)))
            
            for i, target_idx in enumerate(target_indices):
                dummy_prediction[:, target_idx] = prediction_reshaped[:, i]
            
            prediction_unscaled = self.scaler.inverse_transform(dummy_prediction)
            
            return prediction_unscaled[:, target_indices]
            
        except Exception as e:
            logger.error(f"Erro durante predição: {e}")
            return None
    
    def _plot_training_history(self, history):
        """Salva gráfico do histórico de treinamento"""
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Treino')
            plt.plot(history.history['val_loss'], label='Validação')
            plt.title('Histórico de Loss')
            plt.xlabel('Épocas')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Treino')
            plt.plot(history.history['val_mae'], label='Validação')
            plt.title('Histórico de MAE')
            plt.xlabel('Épocas')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Gráfico de treinamento salvo: lstm_training_history.png")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar gráfico: {e}")

class SentinelaVerde:
    """Classe principal do sistema Sentinela Verde"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Inicializar componentes
        self.validator = DataValidator(self.config['sensor_ranges'])
        self.classifier = AirQualityClassifier(self.config['air_quality_limits'])
        self.lstm_predictor = LSTMPredictor(self.config['lstm']) if self.config['lstm']['enabled'] else None
        
        self.df_data = None
        self.df_classified = None
        
    def load_data(self) -> bool:
        """Carrega e valida dados de entrada"""
        input_file = self.config['files']['input_csv']
        
        try:
            # Carregar CSV
            self.df_data = pd.read_csv(input_file)
            logger.info(f"Arquivo '{input_file}' carregado: {len(self.df_data)} registros")
            
            # Verificar colunas obrigatórias
            expected_columns = self.config['columns']['sensors']
            missing_columns = [col for col in expected_columns if col not in self.df_data.columns]
            
            if missing_columns:
                logger.warning(f"Colunas esperadas não encontradas: {missing_columns}")
            
            # Processar timestamp
            timestamp_col = self.config['columns']['timestamp']
            if timestamp_col and timestamp_col in self.df_data.columns:
                try:
                    self.df_data[timestamp_col] = pd.to_datetime(self.df_data[timestamp_col])
                    self.df_data.set_index(timestamp_col, inplace=True)
                    logger.info(f"Coluna '{timestamp_col}' processada como índice temporal")
                except Exception as e:
                    logger.warning(f"Erro ao processar timestamp: {e}")
            
            # Validar dados
            self.df_data, validation_stats = self.validator.validate_sensor_data(self.df_data)
            
            # Detectar anomalias
            self.df_data = self.validator.detect_anomalies(
                self.df_data, 
                self.config['columns']['sensors']
            )
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {input_file}")
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False
    
    def classify_air_quality(self) -> bool:
        """Classifica qualidade do ar"""
        if self.df_data is None:
            logger.error("Dados não carregados")
            return False
        
        try:
            logger.info("Iniciando classificação da qualidade do ar...")
            self.df_classified = self.classifier.calculate_air_quality(self.df_data)
            
            # Salvar dados classificados
            output_file = self.config['files']['output_classified_csv']
            self.df_classified.reset_index().to_csv(output_file, index=False)
            logger.info(f"Dados classificados salvos em: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            return False
    
    def train_and_predict(self) -> bool:
        """Treina modelo LSTM e faz predições"""
        if not self.config['lstm']['enabled'] or self.lstm_predictor is None:
            logger.info("LSTM desabilitado")
            return True
        
        if self.df_classified is None:
            logger.error("Dados não classificados")
            return False
        
        try:
            # Treinar modelo
            if not self.lstm_predictor.train(self.df_classified, self.config['columns']['sensors']):
                logger.error("Falha no treinamento LSTM")
                return False
            
            # Fazer predições
            predictions = self.lstm_predictor.predict_future(self.df_classified)
            
            if predictions is not None:
                self._save_dashboard_data(predictions)
                return True
            else:
                logger.error("Falha na predição")
                return False
                
        except Exception as e:
            logger.error(f"Erro no LSTM: {e}")
            return False
    
    def _save_dashboard_data(self, predictions: np.ndarray):
        """Salva dados para dashboard"""
        try:
            # Última leitura
            last_reading = self.df_classified.iloc[-1]
            timestamp_last = last_reading.name if hasattr(last_reading, 'name') else datetime.now()
            
            if not isinstance(timestamp_last, datetime):
                timestamp_last = datetime.now()
            
            # Timestamps futuros
            future_timestamps = pd.date_range(
                start=timestamp_last + timedelta(hours=1),
                periods=self.config['lstm']['prediction_horizon'],
                freq='H'
            )
            
            # Estrutura do JSON
            dashboard_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'model_version': '2.0',
                    'prediction_horizon_hours': self.config['lstm']['prediction_horizon']
                },
                'ultima_leitura_registrada': {
                    'timestamp': timestamp_last.isoformat(),
                    'valores': {
                        col: round(float(last_reading[col]), 5) 
                        if pd.notna(last_reading[col]) and isinstance(last_reading[col], (int, float, np.number))
                        else str(last_reading[col])
                        for col in self.config['columns']['sensors'] 
                        if col in last_reading
                    },
                    'qualidade_ar': {
                        'categoria': last_reading.get('Qualidade_Ar_Calculada', 'N/A'),
                        'max_subindice': round(float(last_reading.get('Max_SubIndice', 0)), 4),
                        'risco_saude': last_reading.get('Risco_Saude', 'N/A')
                    }
                },
                'previsoes_futuras': []
            }
            
            # Adicionar predições
            for i, timestamp in enumerate(future_timestamps):
                prediction_values = {}
                for j, target_col in enumerate(self.config['lstm']['target_columns']):
                    prediction_values[f'{target_col}_previsto'] = round(float(predictions[i, j]), 5)
                
                dashboard_data['previsoes_futuras'].append({
                    'timestamp': timestamp.isoformat(),
                    'valores_previstos': prediction_values
                })
            
            # Salvar JSON
            output_file = self.config['files']['output_prediction_json']
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados do dashboard salvos em: {output_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados do dashboard: {e}")
    
    def run(self) -> bool:
        """Executa pipeline completo"""
        logger.info("=== Iniciando Sentinela Verde ===")
        
        try:
            # 1. Carregar dados
            if not self.load_data():
                return False
            
            # 2. Classificar qualidade do ar
            if not self.classify_air_quality():
                return False
            
            # 3. Treinar e fazer predições (se habilitado)
            if not self.train_and_predict():
                return False
            
            logger.info("=== Sentinela Verde finalizado com sucesso ===")
            return True
            
        except Exception as e:
            logger.error(f"Erro crítico: {e}")
            return False

def main():
    """Função principal"""
    try:
        # Inicializar sistema
        sentinela = SentinelaVerde()
        
        # Executar pipeline
        success = sentinela.run()
        
        if success:
            print("\n✅ Sistema executado com sucesso!")
            print("📊 Verifique os arquivos gerados:")
            print(f"   - {sentinela.config['files']['output_classified_csv']}")
            print(f"   - {sentinela.config['files']['output_prediction_json']}")
            print("📈 Gráficos salvos como PNG")
        else:
            print("\n❌ Execução finalizada com erros. Verifique o log.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Execução interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro crítico na main: {e}")
        print(f"\n💥 Erro crítico: {e}")

if __name__ == "__main__":
    main()