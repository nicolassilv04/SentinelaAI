# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys

# Imports for Decision Tree and Gas Forecasting
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Configurar warnings e logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # To ignore some statsmodels warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinela_verde.log', mode='w'), # Overwrite log file each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Gerenciador de configurações do sistema"""
    
    DEFAULT_CONFIG = {
        'files': {
            'input_csv': 'meus_dados_arduino.csv', # Example input file
            'output_classified_csv': 'dados_arduino_classificados_regras.csv', # Output from rule-based classifier
            'config_file': 'config.yaml'
        },
        'columns': {
            'timestamp': 'Timestamp', # Ensure your CSV has a timestamp column with this name or configure it
            'sensors': [
                'Amonia_ppm',
                'Benzeno_ppm', 
                'Alcool_ppm',
                'Dioxido_Carbono_ppm',
                'Temperatura_C',
                'Umidade_Relativa_percent'
            ],
        },
        'air_quality_limits': { 
            'Amonia_ppm': 1.0,        
            'Benzeno_ppm': 0.05,      
            'Alcool_ppm': 2.0,    
            'Dioxido_Carbono_ppm': 2000  
        },
        'sensor_ranges': { # Physical limits of sensors for validation
            'Amonia_ppm': {'min': 0, 'max': 50},
            'Benzeno_ppm': {'min': 0, 'max': 10},
            'Alcool_ppm': {'min': 0, 'max': 100},
            'Dioxido_Carbono_ppm': {'min': 0, 'max': 5000},
            'Temperatura_C': {'min': -40, 'max': 85},
            'Umidade_Relativa_percent': {'min': 0, 'max': 100}
        },
        'decision_tree': {
            'enabled': True,
            'min_data_points_tree': 30, # Minimum samples to attempt training
            'test_size': 0.25,
            'random_state': 42,
            'criterion': 'gini', # 'gini' or 'entropy'
            'max_depth': 10, # Limit depth to prevent overfitting and for better visualization
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'plot_figsize': [25, 15], # Adjusted for potentially larger trees
            'plot_fontsize': 8,
            'plot_path': 'decision_tree_air_quality.png',
            'feature_columns': [ # Features for the decision tree
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm',
                'Temperatura_C', 'Umidade_Relativa_percent'
            ],
            'target_column': 'Qualidade_Ar_Calculada' # From AirQualityClassifier
        },
        'gas_forecasting': {
            'enabled': True,
            'prediction_horizon_hours': 24,
            'target_gas_columns': [ # Gases to forecast
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'
            ],
            'min_data_for_train': 50 # Minimum data points for training ETS models
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
                # Deep merge user config with default config
                return self._merge_configs(self.DEFAULT_CONFIG.copy(), loaded_config)
            except Exception as e:
                logger.warning(f"Erro ao carregar configuração de {self.config_path}: {e}. Usando configuração padrão.")
        else:
            logger.info(f"Arquivo de configuração '{self.config_path}' não encontrado. Criando e usando configuração padrão.")
            self.save_default_config()
        return self.DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merges loaded configuration into default configuration."""
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
            logger.error(f"Erro ao salvar configuração padrão: {e}")

class DataValidator:
    """Validador de dados dos sensores"""
    def __init__(self, sensor_ranges: Dict[str, Dict[str, float]]):
        self.sensor_ranges = sensor_ranges
    
    def validate_sensor_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        validation_stats = {}
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if column in self.sensor_ranges:
                initial_count = len(df_clean)
                sensor_range = self.sensor_ranges[column]
                
                # Convert to numeric, coercing errors. This helps if data is read as object.
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                
                mask = (df_clean[column] >= sensor_range['min']) & (df_clean[column] <= sensor_range['max'])
                # Keep NaNs introduced by coerce for now, handle them later or let them be excluded by mask
                df_clean = df_clean[mask | df_clean[column].isnull()] 
                
                removed_count = initial_count - len(df_clean[mask]) # Count only valid removals
                validation_stats[column] = removed_count
                
                if removed_count > 0:
                    logger.warning(f"Removidos {removed_count} valores fora do range físico da coluna {column}")
        
        # Optionally, handle NaNs more explicitly here, e.g., by imputation or removal
        # For now, let them propagate; subsequent steps might handle them.
        return df_clean, validation_stats
    
    def detect_anomalies(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_clean = df.copy()
        for column in columns:
            if column in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[column]):
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0: # Avoid division by zero or issues with constant data
                    continue 
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                outliers_count = outliers_mask.sum()
                if outliers_count > 0:
                    logger.info(f"Detectados {outliers_count} outliers (IQR) em {column}. Eles não são removidos por esta função.")
                    df_clean[f'{column}_is_outlier_IQR'] = outliers_mask # Mark outliers
        return df_clean

class AirQualityClassifier:
    """Classificador de qualidade do ar (baseado em regras/índices)"""

    def __init__(self, limits: Dict[str, float]):
        self.limits = limits
        self.categories = { 
            (0, 0.3): "Excelente",      
            (0.3, 0.6): "Bom",          
            (0.6, 1.0): "Regular",      
            (1.0, 1.5): "Ruim",         
            (1.5, 2.0): "Muito Ruim",   
            (2.0, float('inf')): "Crítico"  
    }
    
    def calculate_air_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        pollutants = [col for col in self.limits.keys() if col in df_result.columns]
        
        if not pollutants:
            logger.error("Nenhum poluente configurado com limites encontrado nos dados para classificação baseada em regras.")
            return df_result
        
        logger.info(f"Calculando qualidade do ar (regras) para: {pollutants}")
        
        sub_indices = pd.DataFrame(index=df_result.index)
        for pollutant in pollutants:
            df_result[pollutant] = pd.to_numeric(df_result[pollutant], errors='coerce')
            # Ensure limit is not zero to avoid division by zero
            limit_value = self.limits.get(pollutant, 1.0)
            if limit_value == 0:
                logger.warning(f"Limite para {pollutant} é 0. Usando 1.0 para evitar divisão por zero no sub-índice.")
                limit_value = 1.0
            sub_indices[f'SubIndice_{pollutant}'] = df_result[pollutant] / limit_value
        
        df_result['Max_SubIndice'] = sub_indices.max(axis=1)
        df_result['Qualidade_Ar_Calculada'] = df_result['Max_SubIndice'].apply(self._categorize)
        df_result['Risco_Saude'] = df_result['Max_SubIndice'].apply(self._assess_health_risk)
        
        self._log_quality_stats(df_result)
        return df_result
    
    def _categorize(self, max_sub_index: float) -> str:
        if pd.isna(max_sub_index): return "Indeterminado"
        for (min_val, max_val), category in self.categories.items():
            if min_val <= max_sub_index < max_val: return category
        return "Crítico" # Default for values >= last max_val
    
    def _assess_health_risk(self, max_sub_index: float) -> str:
        if pd.isna(max_sub_index): return "Indeterminado"
        if max_sub_index <= 0.5: return "Baixo"
        if max_sub_index <= 1.0: return "Moderado"
        if max_sub_index <= 1.5: return "Alto"
        return "Muito Alto"
    
    def _log_quality_stats(self, df: pd.DataFrame):
        if 'Qualidade_Ar_Calculada' in df.columns:
            stats = df['Qualidade_Ar_Calculada'].value_counts(normalize=True) * 100
            logger.info("Distribuição da Qualidade do Ar (Regras):")
            for category, percentage in stats.items():
                logger.info(f"  {category}: {percentage:.1f}%")

    def diagnose_classification(self, df: pd.DataFrame):
        """Método para diagnosticar problemas na classificação"""
        pollutants = [col for col in self.limits.keys() if col in df.columns]
        
        print("\n🔍 DIAGNÓSTICO DA CLASSIFICAÇÃO:")
        print("="*50)
        
        for pollutant in pollutants:
            values = df[pollutant].dropna()
            if len(values) == 0:
                continue
                
            limit = self.limits[pollutant]
            sub_indices = values / limit
            
            print(f"\n{pollutant}:")
            print(f"  Limite configurado: {limit}")
            print(f"  Valores - Min: {values.min():.4f}, Max: {values.max():.4f}, Média: {values.mean():.4f}")
            print(f"  Sub-índices - Min: {sub_indices.min():.2f}, Max: {sub_indices.max():.2f}, Média: {sub_indices.mean():.2f}")
            
            # Contar quantos valores excedem cada threshold
            print(f"  Acima do limite (>1.0): {(sub_indices > 1.0).sum()}/{len(sub_indices)} ({(sub_indices > 1.0).mean()*100:.1f}%)")
            print(f"  Crítico (>2.0): {(sub_indices > 2.0).sum()}/{len(sub_indices)} ({(sub_indices > 2.0).mean()*100:.1f}%)")
        
        if 'Max_SubIndice' in df.columns:
            max_sub = df['Max_SubIndice'].dropna()
            print(f"\nMAX SUB-ÍNDICE GERAL:")
            print(f"  Min: {max_sub.min():.2f}, Max: {max_sub.max():.2f}, Média: {max_sub.mean():.2f}")
            
        print("="*50)

class SimpleGasForecaster:
    """Previsor Simplificado de Concentração de Gases"""
    def __init__(self, config: Dict):
        self.config = config
        self.models = {} # Stores a trained model for each gas
        self.target_gas_columns = config.get('target_gas_columns', [])
        self.min_data_for_train = config.get('min_data_for_train', 50)
        self.trained_on_index_freq = None

    def train(self, df_historical_data: pd.DataFrame) -> bool:
        logger.info("Treinando previsores simples de gases...")
        if df_historical_data.empty:
            logger.warning("Dados históricos vazios para treinar o previsor de gases.")
            return False
        
        # Try to infer frequency if DataFrame index is DatetimeIndex
        if isinstance(df_historical_data.index, pd.DatetimeIndex):
            self.trained_on_index_freq = pd.infer_freq(df_historical_data.index)
            if self.trained_on_index_freq:
                 logger.info(f"Frequência de dados inferida para treino do previsor: {self.trained_on_index_freq}")
            else:
                 logger.warning("Não foi possível inferir a frequência dos dados para o treino do previsor. As previsões podem ser menos precisas.")
        
        for gas_col in self.target_gas_columns:
            if gas_col in df_historical_data.columns:
                series = df_historical_data[gas_col].dropna().astype(float)
                if len(series) >= self.min_data_for_train:
                    try:
                        # Using Holt-Winters Exponential Smoothing
                        # Parameters can be tuned or made configurable
                        model = ExponentialSmoothing(series, trend='add', seasonal='add', 
                                                     seasonal_periods=24 if self.trained_on_index_freq == 'H' or not self.trained_on_index_freq else None, # Adapt seasonal_periods based on freq
                                                     initialization_method='estimated')
                        self.models[gas_col] = model.fit()
                        logger.info(f"Treinado modelo Exponential Smoothing para {gas_col}")
                    except Exception as e:
                        logger.warning(f"Não foi possível treinar Exponential Smoothing para {gas_col}: {e}. Usando persistência (último valor).")
                        self.models[gas_col] = series.iloc[-1] if not series.empty else 0.0
                else:
                    logger.warning(f"Dados insuficientes ({len(series)} < {self.min_data_for_train}) para treinar modelo para {gas_col}. Usando persistência.")
                    self.models[gas_col] = series.iloc[-1] if not series.empty else 0.0
            else:
                logger.warning(f"Coluna alvo de gás {gas_col} não encontrada nos dados para previsão.")
        return bool(self.models)

    def forecast(self, n_periods: int, last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        if not self.models:
            logger.error("Nenhum modelo de previsão de gás treinado.")
            return None

        forecast_data = {}
        
        if last_timestamp is None: # If no last timestamp provided, start from now
            last_timestamp = datetime.now()
        
        # Create future timestamps. Defaulting to Hourly if not inferable.
        freq_to_use = self.trained_on_index_freq if self.trained_on_index_freq else 'H'
        try:
            future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1 if freq_to_use=='H' else 0), periods=n_periods, freq=freq_to_use) # Adjust Timedelta based on freq
        except ValueError as e: # Handle cases where freq might be incompatible with simple addition
            logger.warning(f"Erro ao criar future_index com frequência {freq_to_use}: {e}. Usando frequência horária padrão.")
            future_index = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=n_periods, freq='H')


        for gas_col, model_or_val in self.models.items():
            try:
                if hasattr(model_or_val, 'forecast'): # Check if it's a statsmodels model
                    forecast_values = model_or_val.forecast(n_periods)
                    forecast_data[gas_col] = forecast_values.values
                else: # It's a fallback (last value)
                    logger.info(f"Usando previsão de persistência para {gas_col}")
                    forecast_data[gas_col] = [model_or_val] * n_periods
            except Exception as e:
                logger.error(f"Erro ao prever {gas_col}: {e}. Usando 0 como fallback.")
                forecast_data[gas_col] = [0.0] * n_periods
        
        if not forecast_data: return None
            
        df_forecast = pd.DataFrame(forecast_data, index=future_index)
        # Ensure all configured target gas columns are present
        for gas_col in self.target_gas_columns:
            if gas_col not in df_forecast.columns:
                df_forecast[gas_col] = [0.0] * n_periods
                logger.warning(f"Nenhuma previsão gerada para {gas_col} (pode não ter sido treinada), usando 0.")
        return df_forecast

class DecisionTreePipeline:
    """Pipeline para Árvore de Decisão"""
    def __init__(self, config: Dict):
        self.config = config 
        self.model: Optional[DecisionTreeClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.class_names: List[str] = []
        self.is_trained = False

    def train(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> bool:
        logger.info("Iniciando treinamento da Árvore de Decisão...")
        self.feature_columns = feature_columns
        
        X = df[feature_columns].copy()
        y_raw = df[target_column].copy()

        # Handle NaNs in features and target before training
        X.fillna(X.mean(), inplace=True) # Impute NaNs in features with mean, or choose another strategy
        y_raw.fillna("Indeterminado", inplace=True) # Fill NaNs in target, or drop

        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)

        if len(X) < self.config.get('min_data_points_tree', 30):
            logger.warning(f"Dados insuficientes para Árvore de Decisão: {len(X)} amostras. Necessário: {self.config.get('min_data_points_tree', 30)}")
            return False

        # Check class distribution before stratification
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        stratify_y = None
        if len(unique_classes) > 1 and min_class_count >= 2:
            stratify_y = y
            logger.info(f"Usando estratificação. Classes: {len(unique_classes)}, menor classe: {min_class_count} amostras")
        else:
            logger.warning(f"Não é possível usar estratificação. Classes: {len(unique_classes)}, menor classe: {min_class_count} amostras")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.25), 
                random_state=self.config.get('random_state', 42),
                stratify=stratify_y
            )
        except ValueError as e:
            logger.warning(f"Erro na divisão estratificada: {e}. Tentando sem estratificação.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.25), 
                random_state=self.config.get('random_state', 42)
            )

        self.model = DecisionTreeClassifier(
            criterion=self.config.get('criterion', 'gini'),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            random_state=self.config.get('random_state', 42)
        )
        
        try:
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)
            logger.info(f"Árvore de Decisão treinada. Acurácia no teste: {accuracy:.4f}")
            self._plot_decision_tree()
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Erro durante o treinamento da Árvore de Decisão: {e}", exc_info=True)
            return False

    def predict(self, df_features: pd.DataFrame) -> Optional[List[str]]:
        if not self.model or not self.is_trained:
            logger.error("Modelo de Árvore de Decisão não treinado.")
            return None
        
        missing_cols = [col for col in self.feature_columns if col not in df_features.columns]
        if missing_cols:
            logger.error(f"Colunas de features ausentes nos dados de predição. Esperado: {self.feature_columns}, Ausentes: {missing_cols}")
            return None
        
        X_pred = df_features[self.feature_columns].copy()
        X_pred.fillna(X_pred.mean(), inplace=True) # Impute NaNs consistent with training if any

        try:
            predictions_encoded = self.model.predict(X_pred)
            predictions_decoded = self.label_encoder.inverse_transform(predictions_encoded)
            return predictions_decoded.tolist()
        except Exception as e:
            logger.error(f"Erro durante a predição com Árvore de Decisão: {e}", exc_info=True)
            return None

    def _plot_decision_tree(self):
        if not self.model: return
        try:
            plt.figure(figsize=tuple(self.config.get('plot_figsize', [20,10]))) # Ensure tuple
            plot_tree(
                self.model,
                feature_names=self.feature_columns,
                class_names=self.class_names,
                filled=True,
                rounded=True,
                fontsize=self.config.get('plot_fontsize', 10),
                max_depth=self.config.get('plot_max_depth', 5) # Limit plot depth for readability
            )
            plot_path = self.config.get('plot_path', 'decision_tree.png')
            plt.title("Árvore de Decisão - Qualidade do Ar (Plot Max Depth: 5)")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico da Árvore de Decisão salvo em: {plot_path}")
        except Exception as e:
            logger.warning(f"Erro ao salvar gráfico da Árvore de Decisão: {e}", exc_info=True)

class SentinelaVerde:
    """Classe principal do sistema Sentinela Verde"""
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
        self.df_classified_rules: Optional[pd.DataFrame] = None # Data classified by rules
        
        # Results attributes for frontend
        self.latest_reading_data: Optional[pd.Series] = None
        self.latest_reading_rules_classification: Optional[str] = None
        self.latest_reading_dt_classification: Optional[str] = None
        self.df_future_gases_forecast: Optional[pd.DataFrame] = None
        self.future_quality_predictions_dt: Optional[List[str]] = None
        self.last_timestamp_processed: Optional[datetime] = None

    def load_data(self) -> bool:
        input_file = self.config['files']['input_csv']
        try:
            self.df_data = pd.read_csv(input_file)
            logger.info(f"Arquivo '{input_file}' carregado: {len(self.df_data)} registros")
            
            ts_col = self.config['columns']['timestamp']
            if ts_col and ts_col in self.df_data.columns:
                try:
                    self.df_data[ts_col] = pd.to_datetime(self.df_data[ts_col])
                    self.df_data.set_index(ts_col, inplace=True)
                    self.df_data.sort_index(inplace=True) # Ensure chronological order
                    logger.info(f"Coluna '{ts_col}' processada como índice temporal e ordenada.")
                    if not self.df_data.empty:
                        self.last_timestamp_processed = self.df_data.index[-1]
                except Exception as e:
                    logger.warning(f"Erro ao processar timestamp '{ts_col}': {e}. Continuando sem índice temporal.")
            else:
                logger.warning(f"Coluna timestamp '{ts_col}' não encontrada ou não configurada. Operações temporais podem ser limitadas.")

            self.df_data, _ = self.validator.validate_sensor_data(self.df_data)
            self.df_data = self.validator.detect_anomalies(self.df_data, self.config['columns']['sensors'])
            
            # Handle NaNs after validation - e.g., forward fill for time series
            for col in self.config['columns']['sensors']:
                if col in self.df_data.columns:
                    self.df_data[col] = pd.to_numeric(self.df_data[col], errors='coerce') # Ensure numeric
            self.df_data.ffill(inplace=True) # Forward fill NaNs
            self.df_data.bfill(inplace=True) # Backward fill remaining NaNs at the beginning

            return True
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {input_file}")
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}", exc_info=True)
        return False

    def classify_air_quality_rules(self) -> bool:
        if self.df_classified_rules is not None:
            self.rule_classifier.diagnose_classification(self.df_classified_rules)

        if self.df_data is None: logger.error("Dados não carregados para classificação por regras."); return False
        try:
            logger.info("Iniciando classificação da qualidade do ar (baseada em regras)...")
            self.df_classified_rules = self.rule_classifier.calculate_air_quality(self.df_data)
            
            output_file = self.config['files']['output_classified_csv']
            self.df_classified_rules.reset_index().to_csv(output_file, index=False) # Save with index if it's timestamp
            logger.info(f"Dados classificados (regras) salvos em: {output_file}")
            
            if not self.df_classified_rules.empty:
                 self.latest_reading_data = self.df_classified_rules.iloc[-1].copy()
                 self.latest_reading_rules_classification = self.latest_reading_data.get('Qualidade_Ar_Calculada')
            return True
        except Exception as e:
            logger.error(f"Erro na classificação por regras: {e}", exc_info=True)
        return False

  # Adicione este código ao final do seu arquivo fixed_app.py

# Completar o método process_models_and_forecasts()
    def process_models_and_forecasts(self) -> bool:
        if self.df_classified_rules is None:
            logger.error("Dados não classificados por regras. Execute classify_air_quality_rules primeiro.")
            return False

        # 1. Train/Use Decision Tree
        if self.decision_tree_pipeline:
            dt_conf = self.config.get('decision_tree', {})
            feature_cols = dt_conf.get('feature_columns', [])
            target_col = dt_conf.get('target_column', 'Qualidade_Ar_Calculada')

            if target_col not in self.df_classified_rules.columns:
                logger.error(f"Coluna alvo '{target_col}' para Árvore de Decisão não encontrada.")
                return False
            
            missing_features = [col for col in feature_cols if col not in self.df_classified_rules.columns]
            if missing_features:
                logger.error(f"Colunas de feature para Árvore de Decisão não encontradas: {missing_features}")
                return False

            # Train DT if not already trained
            if not self.decision_tree_pipeline.is_trained:
                df_for_dt_train = self.df_classified_rules.dropna(subset=[target_col]).copy()
                
                if len(df_for_dt_train) < dt_conf.get('min_data_points_tree', 30):
                    logger.warning("Dados insuficientes para treinar Árvore de Decisão")
                else:
                    success = self.decision_tree_pipeline.train(df_for_dt_train, feature_cols, target_col)
                    if success:
                        logger.info("Árvore de Decisão treinada com sucesso")
                    else:
                        logger.error("Falha no treinamento da Árvore de Decisão")

            # Make prediction for latest reading
            if self.decision_tree_pipeline.is_trained and self.latest_reading_data is not None:
                latest_features = pd.DataFrame([self.latest_reading_data[feature_cols]])
                predictions = self.decision_tree_pipeline.predict(latest_features)
                if predictions:
                    self.latest_reading_dt_classification = predictions[0]

        # 2. Gas Forecasting
        if self.gas_forecaster:
            gf_conf = self.config.get('gas_forecasting', {})
            
            # Train forecaster
            if self.gas_forecaster.train(self.df_classified_rules):
                logger.info("Modelos de previsão de gases treinados")
                
                # Generate forecasts
                prediction_horizon = gf_conf.get('prediction_horizon_hours', 24)
                last_ts = self.last_timestamp_processed
                
                self.df_future_gases_forecast = self.gas_forecaster.forecast(
                    prediction_horizon, last_ts
                )
                
                if self.df_future_gases_forecast is not None:
                    logger.info(f"Previsões de gases geradas para {prediction_horizon} períodos")
                    
                    # Classify future air quality using rules
                    future_classified = self.rule_classifier.calculate_air_quality(
                        self.df_future_gases_forecast.copy()
                    )
                    
                    # Predict future air quality using Decision Tree if available
                    if self.decision_tree_pipeline and self.decision_tree_pipeline.is_trained:
                        dt_conf = self.config.get('decision_tree', {})
                        feature_cols = dt_conf.get('feature_columns', [])
                        
                        available_features = [col for col in feature_cols 
                                            if col in self.df_future_gases_forecast.columns]
                        
                        if available_features:
                            self.future_quality_predictions_dt = self.decision_tree_pipeline.predict(
                                self.df_future_gases_forecast[available_features]
                            )

        return True

    def run_analysis(self) -> bool:
        """Executa toda a análise do sistema"""
        logger.info("=== INICIANDO ANÁLISE SENTINELA VERDE ===")
        
        # 1. Carregar dados
        if not self.load_data():
            logger.error("Falha ao carregar dados. Abortando análise.")
            return False
        
        # 2. Classificar qualidade do ar por regras
        if not self.classify_air_quality_rules():
            logger.error("Falha na classificação por regras. Abortando análise.")
            return False
        
        # 3. Processar modelos e previsões
        if not self.process_models_and_forecasts():
            logger.error("Falha no processamento de modelos. Continuando...")
        
        # 4. Exibir resultados
        self.display_results()
        
        logger.info("=== ANÁLISE CONCLUÍDA ===")
        return True

    def display_results(self):
        """Exibe os resultados da análise no console"""
        print("\n" + "="*60)
        print("           RELATÓRIO SENTINELA VERDE")
        print("="*60)
        
        if self.latest_reading_data is not None:
            print(f"\n📊 ÚLTIMA LEITURA:")
            print(f"   Timestamp: {self.last_timestamp_processed}")
            
            # Mostrar valores dos sensores
            sensor_cols = self.config['columns']['sensors']
            for sensor in sensor_cols:
                if sensor in self.latest_reading_data:
                    value = self.latest_reading_data[sensor]
                    print(f"   {sensor}: {value:.2f}")
            
            print(f"\n🌬️  QUALIDADE DO AR:")
            print(f"   Classificação (Regras): {self.latest_reading_rules_classification}")
            
            if hasattr(self, 'latest_reading_dt_classification') and self.latest_reading_dt_classification:
                print(f"   Classificação (IA): {self.latest_reading_dt_classification}")
            
            if 'Risco_Saude' in self.latest_reading_data:
                print(f"   Risco à Saúde: {self.latest_reading_data['Risco_Saude']}")
        
        if self.df_future_gases_forecast is not None:
            print(f"\n🔮 PREVISÕES (Próximas {len(self.df_future_gases_forecast)} horas):")
            target_gases = self.config.get('gas_forecasting', {}).get('target_gas_columns', [])
            
            for gas in target_gases:
                if gas in self.df_future_gases_forecast.columns:
                    avg_forecast = self.df_future_gases_forecast[gas].mean()
                    max_forecast = self.df_future_gases_forecast[gas].max()
                    print(f"   {gas}: Média {avg_forecast:.2f}, Máximo {max_forecast:.2f}")
        
        print("\n📁 ARQUIVOS GERADOS:")
        if hasattr(self, 'df_classified_rules') and self.df_classified_rules is not None:
            print(f"   • {self.config['files']['output_classified_csv']}")
        
        dt_config = self.config.get('decision_tree', {})
        if dt_config.get('enabled') and 'plot_path' in dt_config:
            print(f"   • {dt_config['plot_path']}")
        
        print(f"   • sentinela_verde.log")
        print("="*60)


def create_sample_data():
    """Cria dados de exemplo para teste se o arquivo não existir"""
    filename = 'meus_dados_arduino.csv'
    if not Path(filename).exists():
        logger.info(f"Criando arquivo de dados de exemplo: {filename}")
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
        n_records = len(dates)
        
        np.random.seed(42)
        
        # Simular condições variadas ao longo do tempo
        base_quality = np.random.choice(['boa', 'regular', 'ruim'], n_records, 
                                       p=[0.6, 0.3, 0.1])  # 60% boa, 30% regular, 10% ruim
        
        data = {
            'Timestamp': dates,
            'Amonia_ppm': [],
            'Benzeno_ppm': [],
            'Alcool_ppm': [],
            'Dioxido_Carbono_ppm': [],
            'Temperatura_C': np.random.normal(25, 5, n_records).clip(-10, 50),
            'Umidade_Relativa_percent': np.random.normal(60, 15, n_records).clip(20, 90)
        }
        
        # Gerar dados baseados na qualidade pretendida
        for i, quality in enumerate(base_quality):
            if quality == 'boa':
                data['Amonia_ppm'].append(np.random.normal(0.2, 0.1))
                data['Benzeno_ppm'].append(np.random.normal(0.01, 0.005))
                data['Alcool_ppm'].append(np.random.normal(0.5, 0.2))
                data['Dioxido_Carbono_ppm'].append(np.random.normal(600, 100))
            elif quality == 'regular':
                data['Amonia_ppm'].append(np.random.normal(0.7, 0.2))
                data['Benzeno_ppm'].append(np.random.normal(0.03, 0.01))
                data['Alcool_ppm'].append(np.random.normal(1.2, 0.3))
                data['Dioxido_Carbono_ppm'].append(np.random.normal(1200, 200))
            else:  # ruim
                data['Amonia_ppm'].append(np.random.normal(1.2, 0.3))
                data['Benzeno_ppm'].append(np.random.normal(0.06, 0.02))
                data['Alcool_ppm'].append(np.random.normal(2.5, 0.5))
                data['Dioxido_Carbono_ppm'].append(np.random.normal(1800, 300))
        
        # Aplicar limites físicos
        data['Amonia_ppm'] = np.clip(data['Amonia_ppm'], 0, 5)
        data['Benzeno_ppm'] = np.clip(data['Benzeno_ppm'], 0, 0.2)
        data['Alcool_ppm'] = np.clip(data['Alcool_ppm'], 0, 5)
        data['Dioxido_Carbono_ppm'] = np.clip(data['Dioxido_Carbono_ppm'], 300, 3000)
        
        df_sample = pd.DataFrame(data)
        df_sample.to_csv(filename, index=False)
        logger.info(f"Arquivo de exemplo criado com {n_records} registros - distribuição mais realista")


def main():
    """Função principal do sistema"""
    try:
        print("Iniciando Sistema Sentinela Verde...")
        
        # Criar dados de exemplo se necessário
        create_sample_data()
        
        # Inicializar sistema
        sentinela = SentinelaVerde()
        
        # Executar análise completa
        success = sentinela.run_analysis()
        
        if success:
            print("\n✅ Sistema executado com sucesso!")
        else:
            print("\n❌ Sistema executado com erros. Verifique o log.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal no sistema: {e}", exc_info=True)
        print(f"\n❌ Erro fatal: {e}")


if __name__ == "__main__":
    main()