# backend.py (versão com get_formatted_summary e log aprimorado)
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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


class ConfigManager:
    DEFAULT_CONFIG = {
        'files': {'input_csv': 'meus_dados_arduino.csv',
                  'output_classified_csv': 'dados_arduino_classificados_regras.csv', 'config_file': 'config.yaml'},
        'columns': {'timestamp': 'Timestamp',
                    'sensors': ['Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm', 'Temperatura_C',
                                'Umidade_Relativa_percent']},
        'air_quality_limits': {'Amonia_ppm': 1.0, 'Benzeno_ppm': 0.05, 'Alcool_ppm': 2.0, 'Dioxido_Carbono_ppm': 2000},
        'sensor_ranges': {'Amonia_ppm': {'min': 0, 'max': 50}, 'Benzeno_ppm': {'min': 0, 'max': 10},
                          'Alcool_ppm': {'min': 0, 'max': 100}, 'Dioxido_Carbono_ppm': {'min': 0, 'max': 5000},
                          'Temperatura_C': {'min': -40, 'max': 85}, 'Umidade_Relativa_percent': {'min': 0, 'max': 100}},
        'decision_tree': {'enabled': True, 'min_data_points_tree': 30, 'test_size': 0.25, 'random_state': 42,
                          'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3,
                          'plot_figsize': [20, 12], 'plot_fontsize': 8, 'plot_path': 'decision_tree_air_quality.png',
                          'feature_columns': ['Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm',
                                              'Temperatura_C', 'Umidade_Relativa_percent'],
                          'target_column': 'Qualidade_Ar_Calculada'},
        'gas_forecasting': {'enabled': True, 'prediction_horizon_hours': 24,
                            'target_gas_columns': ['Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'],
                            'min_data_for_train': 50},
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
        except Exception as e:
            logger.error(f"Erro ao salvar config padrão: {e}")


class DataValidator:
    def __init__(self, sensor_ranges: Dict[str, Dict[str, float]]):
        self.sensor_ranges = sensor_ranges

    def validate_sensor_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        df_clean = df.copy()
        for column in df_clean.columns:
            if column in self.sensor_ranges:
                sensor_range = self.sensor_ranges[column]
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                df_clean = df_clean[
                    (df_clean[column] >= sensor_range['min']) & (df_clean[column] <= sensor_range['max']) | df_clean[
                        column].isnull()]
        return df_clean, {}

    def detect_anomalies(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return df


class AirQualityClassifier:
    def __init__(self, limits: Dict[str, float]):
        self.limits = limits
        self.categories = {(0, 0.3): "Excelente", (0.3, 0.6): "Bom", (0.6, 1.0): "Regular", (1.0, 1.5): "Ruim",
                           (1.5, 2.0): "Muito Ruim", (2.0, float('inf')): "Crítico"}

    def calculate_air_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        pollutants = [col for col in self.limits.keys() if col in df_result.columns]
        if not pollutants: return df_result
        sub_indices = pd.DataFrame(index=df_result.index)
        for pollutant in pollutants:
            df_result[pollutant] = pd.to_numeric(df_result[pollutant], errors='coerce')
            limit_value = self.limits.get(pollutant, 1.0)
            if limit_value == 0: limit_value = 1.0
            sub_indices[f'SubIndice_{pollutant}'] = df_result[pollutant] / limit_value
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
    def __init__(self, config: Dict):
        self.config = config;
        self.models = {};
        self.target_gas_columns = config.get('target_gas_columns', []);
        self.min_data_for_train = config.get('min_data_for_train', 50);
        self.trained_on_index_freq = None

    def train(self, df_historical_data: pd.DataFrame) -> bool:
        if isinstance(df_historical_data.index, pd.DatetimeIndex): self.trained_on_index_freq = pd.infer_freq(
            df_historical_data.index)
        for gas_col in self.target_gas_columns:
            if gas_col in df_historical_data.columns:
                series = df_historical_data[gas_col].dropna().astype(float)
                if len(series) >= self.min_data_for_train:
                    try:
                        self.models[gas_col] = ExponentialSmoothing(series, trend='add', seasonal='add',
                                                                    seasonal_periods=24 if self.trained_on_index_freq == 'H' else None,
                                                                    initialization_method="estimated").fit()
                        logger.info(f"Modelo Sazonal treinado com sucesso para {gas_col}.")
                    except Exception as e1:
                        logger.warning(
                            f"Falha ao treinar modelo sazonal para {gas_col}: {e1}. Tentando modelo mais simples...")
                        try:
                            self.models[gas_col] = ExponentialSmoothing(series, trend='add', seasonal=None,
                                                                        initialization_method="estimated").fit()
                            logger.info(f"Modelo Linear (Holt) treinado com sucesso para {gas_col}.")
                        except Exception as e2:
                            logger.error(
                                f"Falha ao treinar modelo simples para {gas_col}: {e2}. Usando último valor como fallback.")
                            self.models[gas_col] = series.iloc[-1] if not series.empty else 0.0
                elif not series.empty:
                    self.models[gas_col] = series.iloc[-1]
                else:
                    self.models[gas_col] = 0.0
        return bool(self.models)

    def forecast(self, n_periods: int, last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        if not self.models: return None
        forecast_data = {};
        last_ts = last_timestamp or datetime.now();
        freq = self.trained_on_index_freq or 'H';
        future_index = pd.date_range(start=last_ts + timedelta(hours=1), periods=n_periods, freq=freq)
        for gas_col, model in self.models.items():
            if hasattr(model, 'forecast'):
                forecast_data[gas_col] = model.forecast(n_periods).values
            else:
                forecast_data[gas_col] = [model] * n_periods
        df_forecast = pd.DataFrame(forecast_data, index=future_index)
        for col in self.target_gas_columns:
            if col not in df_forecast.columns: df_forecast[col] = 0.0
        return df_forecast


class DecisionTreePipeline:
    def __init__(self, config: Dict):
        self.config = config;
        self.model: Optional[DecisionTreeClassifier] = None;
        self.label_encoder = LabelEncoder();
        self.feature_columns: List[str] = [];
        self.class_names: List[str] = [];
        self.is_trained = False

    def train(self, df: pd.DataFrame, feature_columns: List[str], target_column: str, **kwargs) -> bool:
        self.feature_columns = feature_columns;
        df_clean = df.copy();
        df_clean[target_column] = df_clean[target_column].fillna("Indeterminado")
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]): df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        X = df_clean[feature_columns];
        y_raw = df_clean[target_column];
        y = self.label_encoder.fit_transform(y_raw);
        self.class_names = list(self.label_encoder.classes_)
        if len(X) < self.config.get('min_data_points_tree', 30): logger.warning(
            f"Dados insuficientes para treinar a Árvore de Decisão: {len(X)} amostras."); return False
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.get('test_size', 0.25),
                                                                random_state=self.config.get('random_state', 42),
                                                                stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.get('test_size', 0.25),
                                                                random_state=self.config.get('random_state', 42))
        dt_params = {k: v for k, v in self.config.items() if
                     k in ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']};
        self.model = DecisionTreeClassifier(**dt_params);
        self.model.fit(X_train, y_train);
        self.is_trained = True;
        logger.info(f"Árvore de Decisão treinada. Acurácia: {self.model.score(X_test, y_test):.4f}");
        self._plot_decision_tree()
        return True

    def predict(self, df_features: pd.DataFrame) -> Optional[List[str]]:
        if not self.is_trained: return None
        X_pred = df_features[self.feature_columns].copy()
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X_pred[col]): X_pred[col] = X_pred[col].fillna(X_pred[col].mean())
        preds = self.model.predict(X_pred)
        return self.label_encoder.inverse_transform(preds).tolist()

    def _plot_decision_tree(self):
        if not self.model: logger.warning("Modelo não treinado, não é possível plotar a árvore."); return
        try:
            figsize = tuple(self.config.get('plot_figsize', (20, 12)));
            fig, ax = plt.subplots(figsize=figsize)
            plot_tree(self.model, feature_names=self.feature_columns, class_names=self.class_names, filled=True,
                      rounded=True, fontsize=self.config.get('plot_fontsize', 10), max_depth=5, ax=ax)
            ax.set_title("Árvore de Decisão - Qualidade do Ar (Visualização)", fontsize=16);
            fig.tight_layout();
            plot_path = self.config.get('plot_path', 'decision_tree_air_quality.png');
            fig.savefig(plot_path, dpi=300);
            plt.close(fig)
            logger.info(f"Gráfico da Árvore de Decisão salvo corretamente em: {plot_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar gráfico da Árvore de Decisão: {e}", exc_info=True)


class SentinelaVerde:
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path);
        self.config = self.config_manager.config;
        self.validator = DataValidator(self.config['sensor_ranges']);
        self.rule_classifier = AirQualityClassifier(self.config['air_quality_limits']);
        dt_config = self.config.get('decision_tree', {});
        self.decision_tree_pipeline = DecisionTreePipeline(dt_config) if dt_config.get('enabled') else None;
        gf_config = self.config.get('gas_forecasting', {});
        self.gas_forecaster = SimpleGasForecaster(gf_config) if gf_config.get('enabled') else None;
        self.df_data: Optional[pd.DataFrame] = None;
        self.df_classified_rules: Optional[pd.DataFrame] = None;
        self.latest_reading_data: Optional[pd.Series] = None;
        self.latest_reading_rules_classification: Optional[str] = None;
        self.latest_reading_dt_classification: Optional[str] = None;
        self.df_future_gases_forecast: Optional[pd.DataFrame] = None;
        self.future_quality_predictions_dt: Optional[List[str]] = None;
        self.last_timestamp_processed: Optional[datetime] = None

    def load_data(self) -> bool:
        input_file = self.config['files']['input_csv']
        try:
            self.df_data = pd.read_csv(input_file);
            ts_col = self.config['columns']['timestamp']
            if ts_col in self.df_data.columns:
                self.df_data[ts_col] = pd.to_datetime(self.df_data[ts_col], errors='coerce');
                self.df_data.dropna(subset=[ts_col], inplace=True)
                self.df_data.set_index(ts_col, inplace=True);
                self.df_data.sort_index(inplace=True)
                if not self.df_data.empty: self.last_timestamp_processed = self.df_data.index[-1]
            self.df_data, _ = self.validator.validate_sensor_data(self.df_data);
            self.df_data.ffill(inplace=True);
            self.df_data.bfill(inplace=True)
            return True
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {input_file}"); return False
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}", exc_info=True); return False

    def classify_air_quality_rules(self) -> bool:
        if self.df_data is None: return False
        self.df_classified_rules = self.rule_classifier.calculate_air_quality(self.df_data)
        if not self.df_classified_rules.empty: self.latest_reading_data = self.df_classified_rules.iloc[
            -1]; self.latest_reading_rules_classification = self.latest_reading_data.get('Qualidade_Ar_Calculada')
        return True

    def process_models_and_forecasts(self) -> bool:
        if self.df_classified_rules is None: return False
        if self.decision_tree_pipeline:
            dt_params = self.config['decision_tree']
            if self.decision_tree_pipeline.train(self.df_classified_rules,
                                                 **dt_params):  # Passa os parâmetros como keyword arguments
                if self.latest_reading_data is not None:
                    prediction = self.decision_tree_pipeline.predict(pd.DataFrame([self.latest_reading_data]))
                    self.latest_reading_dt_classification = prediction[0] if prediction else "Falha na Predição"
        if self.gas_forecaster and self.gas_forecaster.train(
            self.df_classified_rules): self.df_future_gases_forecast = self.gas_forecaster.forecast(
            self.config['gas_forecasting']['prediction_horizon_hours'], self.last_timestamp_processed)
        return True

    # <<< MÉTODO run_analysis ATUALIZADO >>>
    def run_analysis(self):
        logger.info("=== INICIANDO ANÁLISE SENTINELA VERDE ===");
        if self.load_data() and self.classify_air_quality_rules():
            self.process_models_and_forecasts()

        # Chamada de display_results para o log de console
        self.display_results()
        logger.info("--- ANÁLISE CONCLUÍDA ---")

    # <<< MÉTODO display_results ATUALIZADO (AGORA SÓ PARA LOG) >>>
    def display_results(self):
        logger.info("=" * 60)
        logger.info("           RELATÓRIO SENTINELA VERDE (LOG INTERNO)")  # Mudança de título para clareza
        logger.info("=" * 60)
        if self.latest_reading_data is not None:
            ts_formatado = self.last_timestamp_processed.strftime(
                '%d/%m/%Y %H:%M:%S') if self.last_timestamp_processed else "N/A"
            logger.info(f"ÚLTIMA LEITURA ({ts_formatado}):")
            for sensor in self.config['columns']['sensors']:
                if sensor in self.latest_reading_data: logger.info(
                    f"   {sensor}: {self.latest_reading_data[sensor]:.2f}")
            logger.info(f"QUALIDADE DO AR (Regras): {self.latest_reading_rules_classification}")
            if self.latest_reading_dt_classification: logger.info(
                f"QUALIDADE DO AR (IA): {self.latest_reading_dt_classification}")
        if self.df_future_gases_forecast is not None: logger.info(
            "PREVISÕES (LOG):\n" + str(self.df_future_gases_forecast.head()))

    # <<< NOVO MÉTODO ADICIONADO >>>
    def get_formatted_summary(self) -> str:
        """Retorna uma string formatada com o resumo dos resultados da análise."""
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("📊           RELATÓRIO SENTINELA VERDE (RESUMO)           📊")
        summary_lines.append("=" * 70)

        if self.latest_reading_data is not None:
            ts_formatado = self.last_timestamp_processed.strftime(
                '%d/%m/%Y %H:%M:%S') if self.last_timestamp_processed else "N/A"
            summary_lines.append(f"\n📈 ÚLTIMA LEITURA REGISTRADA ({ts_formatado}):")

            sensor_cols = self.config['columns']['sensors']
            for sensor in sensor_cols:
                if sensor in self.latest_reading_data:
                    value = self.latest_reading_data[sensor]
                    summary_lines.append(f"   - {sensor.replace('_', ' ')}: {value:.2f}")

            summary_lines.append(f"\n🌬️ QUALIDADE DO AR (BASEADA EM REGRAS):")
            summary_lines.append(f"   - Classificação: {self.latest_reading_rules_classification or 'N/A'}")
            if 'Risco_Saude' in self.latest_reading_data:
                summary_lines.append(f"   - Risco à Saúde: {self.latest_reading_data['Risco_Saude'] or 'N/A'}")

            if self.latest_reading_dt_classification:
                summary_lines.append(f"\n🧠 QUALIDADE DO AR (PREVISÃO IA):")
                summary_lines.append(f"   - Classificação: {self.latest_reading_dt_classification}")
        else:
            summary_lines.append("\n⚠️ Nenhuma leitura de dados processada ainda.")

        if self.df_future_gases_forecast is not None and not self.df_future_gases_forecast.empty:
            summary_lines.append(
                f"\n🔮 PREVISÕES DE CONCENTRAÇÃO DE GASES (Próximas {len(self.df_future_gases_forecast)} horas):")
            target_gases = self.config.get('gas_forecasting', {}).get('target_gas_columns', [])

            for gas in target_gases:
                if gas in self.df_future_gases_forecast.columns:
                    avg_forecast = self.df_future_gases_forecast[gas].mean()
                    max_forecast = self.df_future_gases_forecast[gas].max()
                    min_forecast = self.df_future_gases_forecast[gas].min()
                    summary_lines.append(
                        f"   - {gas.replace('_ppm', '')}: Média {avg_forecast:.2f} ppm (Min: {min_forecast:.2f}, Máx: {max_forecast:.2f})")
        else:
            summary_lines.append("\n⚠️ Nenhuma previsão de gases disponível.")

        summary_lines.append("\n" + "=" * 70)
        return "\n".join(summary_lines)


def create_sample_data():
    filename = ConfigManager.DEFAULT_CONFIG['files']['input_csv'];
    limits = ConfigManager.DEFAULT_CONFIG['air_quality_limits']
    logger.info(f"Gerando/Sobrescrevendo arquivo de dados de exemplo variado: {filename}");
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H');
    n_records = len(dates)
    scenarios = np.random.choice(['Bom', 'Regular', 'Crítico'], n_records, p=[0.6, 0.3, 0.1]);
    data = {'Timestamp': dates}
    for gas in limits.keys(): data[gas] = np.zeros(n_records)
    for i, scenario in enumerate(scenarios):
        for gas, limit in limits.items():
            if scenario == 'Bom':
                data[gas][i] = np.random.uniform(0, limit * 0.5)
            elif scenario == 'Regular':
                data[gas][i] = np.random.uniform(limit * 0.7, limit * 1.2)
            else:
                data[gas][i] = np.random.uniform(limit * 1.5, limit * 3.0)
    data['Temperatura_C'] = np.random.normal(22, 5, n_records);
    data['Umidade_Relativa_percent'] = np.random.normal(60, 15, n_records);
    df_sample = pd.DataFrame(data)
    for col, ranges in ConfigManager.DEFAULT_CONFIG['sensor_ranges'].items():
        if col in df_sample.columns: df_sample[col] = df_sample[col].clip(ranges['min'], ranges['max'])
    df_sample.to_csv(filename, index=False, float_format='%.3f');
    logger.info(f"Arquivo '{filename}' gerado com {n_records} registros variados.")


def main():
    print("Iniciando Sistema Sentinela Verde (modo CLI)...");
    create_sample_data();
    sentinela = SentinelaVerde();
    sentinela.run_analysis();
    print("\n✅ Sistema executado com sucesso!")


if __name__ == "__main__":
    main()