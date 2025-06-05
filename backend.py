# backend.py (versão corrigida e sem geração de dados)
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys
import time

# Imports para Modelagem
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Configurações de Logging e Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinela_verde.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Classes Principais do Backend ---

class ConfigManager:
    """Gerencia as configurações do sistema a partir de um arquivo YAML ou usa um padrão."""
    DEFAULT_CONFIG = {
        'files': {
            'input_csv': 'meus_dados_arduino_historico.csv',
            'output_classified_csv': 'dados_arduino_classificados_regras.csv',
            'config_file': 'config.yaml'
        },
        'columns': {
            'timestamp': 'Timestamp',
            'sensors': [
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 
                'Dioxido_Carbono_ppm', 'Temperatura_C', 'Umidade_Relativa_percent'
            ],
        },
        'air_quality_limits': {
            'Amonia_ppm': 1.0, 'Benzeno_ppm': 0.05, 
            'Alcool_ppm': 2.0, 'Dioxido_Carbono_ppm': 2000
        },
        'sensor_ranges': {
            'Amonia_ppm': {'min': 0, 'max': 50}, 'Benzeno_ppm': {'min': 0, 'max': 10},
            'Alcool_ppm': {'min': 0, 'max': 100}, 'Dioxido_Carbono_ppm': {'min': 0, 'max': 5000},
            'Temperatura_C': {'min': -40, 'max': 85}, 'Umidade_Relativa_percent': {'min': 0, 'max': 100}
        },
        'decision_tree': {
            'enabled': True, 'min_data_points_tree': 20,
            'test_size': 0.30, 'random_state': 42, 'criterion': 'entropy', 'max_depth': 10,
            'feature_columns': [
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm',
                'Temperatura_C', 'Umidade_Relativa_percent'
            ],
            'target_column': 'Qualidade_Ar_Calculada',
            'plot_tree_enabled': True,
            'plot_path': 'decision_tree_air_quality.png'
        },
        'gas_forecasting': {
            'enabled': True, 'prediction_horizon_hours': 24,
            'target_gas_columns': [
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'
            ],
            'min_data_for_train': 20
        },
    }

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self.DEFAULT_CONFIG['files']['config_file']
        self.config = self.load_config()

    def load_config(self) -> Dict:
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                logger.info(f"Configuração carregada de {self.config_path}")
                return self._merge_configs(self.DEFAULT_CONFIG.copy(), loaded_config)
            except Exception as e:
                logger.warning(f"Erro ao carregar {self.config_path}: {e}. Usando config padrão.")
        else:
            self.save_default_config()
        return self.DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        for key, value in loaded.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def save_default_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"Configuração padrão salva em {self.config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar config padrão: {e}")


class DataValidator:
    """Valida os dados do sensor com base nos ranges físicos definidos na configuração."""
    def __init__(self, sensor_ranges: Dict[str, Dict[str, float]]):
        self.sensor_ranges = sensor_ranges

    def validate_sensor_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        df_clean = df.copy()
        for column in df_clean.columns:
            if column in self.sensor_ranges:
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                sensor_range = self.sensor_ranges[column]
                
                # Mantém valores dentro do range ou NaN (que será tratado depois)
                mask = (df_clean[column] >= sensor_range['min']) & (df_clean[column] <= sensor_range['max'])
                df_clean = df_clean[mask | df_clean[column].isnull()]
        return df_clean, {}

class AirQualityClassifier:
    """Classifica a qualidade do ar com base em regras de sub-índices."""
    def __init__(self, limits: Dict[str, float]):
        self.limits = limits
        self.categories = {
            (0, 0.3): "Excelente", (0.3, 0.6): "Bom", (0.6, 1.0): "Regular",
            (1.0, 1.5): "Ruim", (1.5, 2.0): "Muito Ruim", (2.0, float('inf')): "Crítico"
        }

    def calculate_air_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        pollutants = [col for col in self.limits.keys() if col in df_result.columns]
        if not pollutants:
            return df_result
            
        sub_indices = pd.DataFrame(index=df_result.index)
        for pollutant in pollutants:
            df_result[pollutant] = pd.to_numeric(df_result[pollutant], errors='coerce')
            limit_value = self.limits.get(pollutant, 1.0)
            sub_indices[f'SubIndice_{pollutant}'] = df_result[pollutant] / (limit_value if limit_value != 0 else 1.0)

        df_result['Max_SubIndice'] = sub_indices.max(axis=1)
        df_result['Qualidade_Ar_Calculada'] = df_result['Max_SubIndice'].apply(self._categorize)
        df_result['Risco_Saude'] = df_result['Max_SubIndice'].apply(self._assess_health_risk)
        return df_result

    def _categorize(self, idx):
        if pd.isna(idx): return "Indeterminado"
        return next((cat for (min_v, max_v), cat in self.categories.items() if min_v <= idx < max_v), "Crítico")

    def _assess_health_risk(self, idx):
        if pd.isna(idx): return "Indeterminado"
        if idx <= 0.5: return "Baixo"
        if idx <= 1.0: return "Moderado"
        if idx <= 1.5: return "Alto"
        return "Muito Alto"


class SimpleGasForecaster:
    """Realiza previsões de séries temporais para as concentrações de gases."""
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.target_gas_columns = config.get('target_gas_columns', [])
        self.min_data_for_train = config.get('min_data_for_train', 50)
        self.trained_on_index_freq = None

    def train(self, df_historical: pd.DataFrame) -> bool:
        if isinstance(df_historical.index, pd.DatetimeIndex):
            self.trained_on_index_freq = pd.infer_freq(df_historical.index)
            
        for gas_col in self.target_gas_columns:
            if gas_col in df_historical.columns:
                series = df_historical[gas_col].dropna().astype(float)
                if len(series) >= self.min_data_for_train:
                    try:
                        seasonal_periods = 24 if self.trained_on_index_freq and 'H' in self.trained_on_index_freq else None
                        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
                        self.models[gas_col] = model
                    except Exception:
                        self.models[gas_col] = series.iloc[-1] # Fallback
                else:
                    self.models[gas_col] = series.iloc[-1] if not series.empty else 0.0
        return bool(self.models)

    def forecast(self, n_periods: int, last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        if not self.models: return None
        
        forecast_data = {}
        last_ts = last_timestamp or datetime.now()
        freq = self.trained_on_index_freq or 'H'
        future_index = pd.date_range(start=last_ts + timedelta(hours=1), periods=n_periods, freq=freq)
        
        for gas_col, model in self.models.items():
            forecast_data[gas_col] = model.forecast(n_periods) if hasattr(model, 'forecast') else [model] * n_periods
            
        return pd.DataFrame(forecast_data, index=future_index)


class DecisionTreePipeline:
    """Pipeline para treinar e usar um modelo de Árvore de Decisão."""
    def __init__(self, config: Dict):
        self.config = config
        self.model: Optional[DecisionTreeClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.class_names: List[str] = []
        self.is_trained = False

    def train(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> bool:
        self.feature_columns = feature_columns
        df_train = df.dropna(subset=[target_column] + feature_columns).copy()

        if len(df_train) < self.config.get('min_data_points_tree', 20):
            logger.warning(f"Dados insuficientes para treinar: {len(df_train)} amostras.")
            return False

        X = df_train[feature_columns]
        y = self.label_encoder.fit_transform(df_train[target_column])
        self.class_names = list(self.label_encoder.classes_)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(self.class_names) > 1 else None)
        
        self.model = DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy')
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"Árvore de Decisão treinada. Acurácia: {self.model.score(X_test, y_test):.4f}")
        if self.config.get('plot_tree_enabled', False):
             self._plot_decision_tree()
        return True

    def predict(self, df_features: pd.DataFrame) -> Optional[List[str]]:
        if not self.is_trained: return None
        X_pred = df_features[self.feature_columns].fillna(df_features[self.feature_columns].mean())
        predictions_encoded = self.model.predict(X_pred)
        return self.label_encoder.inverse_transform(predictions_encoded).tolist()
        
    def _plot_decision_tree(self):
        if not self.model: return
        try:
            plt.figure(figsize=(25, 15))
            plot_tree(self.model, feature_names=self.feature_columns, class_names=self.class_names, filled=True, rounded=True, fontsize=10)
            plt.savefig(self.config.get('plot_path', 'decision_tree.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao plotar árvore: {e}")


class SentinelaVerde:
    """Classe orquestradora principal do sistema."""
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.validator = DataValidator(self.config['sensor_ranges'])
        self.rule_classifier = AirQualityClassifier(self.config['air_quality_limits'])
        
        dt_conf = self.config.get('decision_tree', {})
        self.decision_tree_pipeline = DecisionTreePipeline(dt_conf) if dt_conf.get('enabled', False) else None
        
        gf_conf = self.config.get('gas_forecasting', {})
        self.gas_forecaster = SimpleGasForecaster(gf_conf) if gf_conf.get('enabled', False) else None
        
        self.df_data: Optional[pd.DataFrame] = None
        self.df_classified_rules: Optional[pd.DataFrame] = None
        self.latest_reading_data: Optional[pd.Series] = None
        self.latest_reading_rules_classification: Optional[str] = None
        self.latest_reading_dt_classification: Optional[str] = None
        self.df_future_gases_forecast: Optional[pd.DataFrame] = None
        self.last_timestamp_processed: Optional[datetime] = None

    def load_data(self) -> bool:
        input_file = self.config['files']['input_csv']
        try:
            # Garante que o arquivo exista
            if not Path(input_file).exists():
                logger.warning(f"Arquivo de dados '{input_file}' não encontrado. Criando um vazio.")
                header = [self.config['columns']['timestamp']] + self.config['columns']['sensors']
                pd.DataFrame(columns=header).to_csv(input_file, index=False)

            self.df_data = pd.read_csv(input_file)
            if self.df_data.empty:
                logger.info("Arquivo de dados está vazio.")
                return True

            ts_col = self.config['columns']['timestamp']
            if ts_col in self.df_data.columns:
                self.df_data[ts_col] = pd.to_datetime(self.df_data[ts_col], errors='coerce')
                self.df_data.dropna(subset=[ts_col], inplace=True)
                if not self.df_data.empty:
                    self.df_data.set_index(ts_col, inplace=True).sort_index(inplace=True)
                    self.last_timestamp_processed = self.df_data.index[-1]
            
            self.df_data, _ = self.validator.validate_sensor_data(self.df_data.reset_index())
            if ts_col in self.df_data.columns: # Re-set index
                 self.df_data.set_index(ts_col, inplace=True)

            self.df_data.ffill(inplace=True).bfill(inplace=True) # Preenchimento de NaNs
            return True
        except Exception as e:
            logger.error(f"Erro crítico ao carregar dados: {e}", exc_info=True)
            return False

    def run_analysis(self):
        """Executa o ciclo completo de análise: carregar, classificar, modelar e prever."""
        logger.info("--- INICIANDO CICLO DE ANÁLISE ---")
        if not self.load_data() or self.df_data is None or self.df_data.empty:
            logger.warning("Análise abortada: não foi possível carregar dados ou o arquivo está vazio.")
            return

        self.df_classified_rules = self.rule_classifier.calculate_air_quality(self.df_data)
        if not self.df_classified_rules.empty:
            self.latest_reading_data = self.df_classified_rules.iloc[-1]
            self.latest_reading_rules_classification = self.latest_reading_data.get('Qualidade_Ar_Calculada')

        if self.decision_tree_pipeline:
            dt_conf = self.config['decision_tree']
            if self.decision_tree_pipeline.train(self.df_classified_rules, dt_conf['feature_columns'], dt_conf['target_column']):
                if self.latest_reading_data is not None:
                    pred = self.decision_tree_pipeline.predict(pd.DataFrame([self.latest_reading_data]))
                    self.latest_reading_dt_classification = pred[0] if pred else "Falha na Predição IA"
        
        if self.gas_forecaster and self.gas_forecaster.train(self.df_classified_rules):
            self.df_future_gases_forecast = self.gas_forecaster.forecast(self.config['gas_forecasting']['prediction_horizon_hours'], self.last_timestamp_processed)
        
        logger.info("--- CICLO DE ANÁLISE CONCLUÍDO ---")

    def get_formatted_summary(self) -> str:
        """Retorna uma string formatada com o resumo dos resultados da análise para a UI."""
        if self.latest_reading_data is None:
            return "Nenhuma leitura de dados válida para exibir."

        summary = [f"ÚLTIMA LEITURA ({self.last_timestamp_processed.strftime('%d/%m/%Y %H:%M') if self.last_timestamp_processed else 'N/A'}):"]
        for sensor, value in self.latest_reading_data.items():
            if sensor in self.config['columns']['sensors']:
                summary.append(f"  - {sensor}: {value:.2f}")

        summary.append(f"\nQUALIDADE DO AR (REGRAS): {self.latest_reading_rules_classification or 'N/A'}")
        if self.latest_reading_dt_classification:
            summary.append(f"QUALIDADE DO AR (IA): {self.latest_reading_dt_classification}")

        if self.df_future_gases_forecast is not None:
            summary.append("\nPREVISÕES (Próximas 24h):")
            summary.append(str(self.df_future_gases_forecast.head()))
        
        return "\n".join(summary)