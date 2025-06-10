# backend.py (versão com ranges e limites de gases ajustados para a realidade do MQ-135)
# -*- coding: utf-8 -*

# --- Importações de Bibliotecas Essenciais ---
import pandas as pd  # Para manipulação e análise de dados em DataFrames
import numpy as np  # Para operações numéricas
from datetime import datetime, timedelta  # Para trabalhar com datas e horas
import yaml  # Para ler e escrever arquivos de configuração em formato YAML
import logging  # Para registrar eventos e erros do sistema
import warnings  # Para gerenciar avisos do sistema
from pathlib import Path  # Para lidar com caminhos de arquivos de forma moderna
from typing import Dict, List, Tuple, Optional, Union  # Para anotações de tipo, melhorando a clareza do código
import sys  # Para interagir com o sistema (usado aqui para o logging)
import time  # Para funções relacionadas a tempo

# --- Importações para Modelagem e Machine Learning ---
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Modelo de Árvore de Decisão e sua visualização
from sklearn.model_selection import train_test_split  # Para dividir dados em conjuntos de treino e teste
from sklearn.preprocessing import LabelEncoder  # Para converter rótulos textuais em números
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para rodar em servidores (sem interface gráfica)
import matplotlib.pyplot as plt  # Para criar gráficos e visualizações
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Modelo para previsão de séries temporais

# --- Configurações Iniciais ---
# Ignora avisos específicos para manter o log limpo
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configura o sistema de logging para salvar em arquivo e exibir no console
logging.basicConfig(
    level=logging.INFO,  # Nível mínimo de mensagem a ser registrada (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato da mensagem de log
    handlers=[
        logging.FileHandler('sentinela_verde.log', mode='w', encoding='utf-8'),  # Salva logs no arquivo
        logging.StreamHandler(sys.stdout)  # Exibe logs no console
    ]
)
logger = logging.getLogger(__name__)  # Cria uma instância do logger

# ==============================================================================
# 1. GERENCIAMENTO DE CONFIGURAÇÃO
# ==============================================================================
class ConfigManager:
    """
    Gerencia as configurações do sistema.
    - Carrega configurações de um arquivo 'config.yaml'.
    - Se o arquivo não existir, cria um com valores padrão.
    - Isso torna o sistema flexível, permitindo ajustes sem alterar o código.
    """
    # Dicionário com todas as configurações padrão do sistema
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
        # LIMITES DE QUALIDADE DO AR: baseados em normas de saúde.
        # Um valor acima do limite indica que a qualidade do ar é "Ruim".
        'air_quality_limits': {
            'Amonia_ppm': 25.0,
            'Benzeno_ppm': 1.0,
            'Alcool_ppm': 50.0,
            'Dioxido_Carbono_ppm': 1500.0
        },
        # RANGES DO SENSOR: limites físicos de detecção do sensor MQ-135.
        # Leituras fora desses valores são consideradas inválidas.
        'sensor_ranges': {
            'Amonia_ppm': {'min': 10, 'max': 300},
            'Benzeno_ppm': {'min': 10, 'max': 1000},
            'Alcool_ppm': {'min': 10, 'max': 300},
            'Dioxido_Carbono_ppm': {'min': 10, 'max': 10000},
            'Temperatura_C': {'min': -10, 'max': 50},
            'Umidade_Relativa_percent': {'min': 20, 'max': 90}
        },
        # Configurações do modelo de Machine Learning (Árvore de Decisão)
        'decision_tree': {
            'enabled': True,
            'min_data_points_tree': 20,
            'feature_columns': [ # Variáveis usadas para treinar o modelo
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm',
                'Temperatura_C', 'Umidade_Relativa_percent'
            ],
            'target_column': 'Qualidade_Ar_Calculada', # O que o modelo tentará prever
            'plot_tree_enabled': True, # Se deve gerar uma imagem da árvore
            'plot_path': 'decision_tree_air_quality.png'
        },
        # Configurações do modelo de Previsão de Séries Temporais
        'gas_forecasting': {
            'enabled': True,
            'prediction_horizon_hours': 24, # Quantas horas no futuro prever
            'target_gas_columns': [
                'Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'
            ],
            'min_data_for_train': 20
        },
    }

    def __init__(self, config_path: str = None):
        """Inicializa o gerenciador, definindo o caminho do arquivo de configuração."""
        self.config_path = config_path or self.DEFAULT_CONFIG['files']['config_file']
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Carrega as configurações do arquivo YAML ou usa os padrões."""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                logger.info(f"Configuração carregada de {self.config_path}")
                # Mescla a config padrão com a carregada para garantir que todas as chaves existam
                return self._merge_configs(self.DEFAULT_CONFIG.copy(), loaded_config)
            except Exception as e:
                logger.warning(f"Erro ao carregar {self.config_path}: {e}. Usando config padrão.")
        else:
            # Se o arquivo não existe, salva o padrão e o utiliza
            self.save_default_config()
        return self.DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Função auxiliar para mesclar dicionários de configuração."""
        for key, value in loaded.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def save_default_config(self):
        """Salva a configuração padrão em um arquivo YAML."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"Configuração padrão salva em {self.config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar config padrão: {e}")


# ==============================================================================
# 2. PRÉ-PROCESSAMENTO E VALIDAÇÃO DE DADOS
# ==============================================================================
class DataValidator:
    """
    Valida os dados brutos do sensor.
    - Garante que as leituras estejam dentro dos ranges operacionais do sensor (definidos na config).
    - Remove leituras inválidas ou não numéricas.
    """
    def __init__(self, sensor_ranges: Dict[str, Dict[str, float]]):
        self.sensor_ranges = sensor_ranges

    def validate_sensor_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        df_clean = df.copy()
        for column in df_clean.columns:
            if column in self.sensor_ranges:
                # Converte a coluna para número, forçando erros a se tornarem NaN (Not a Number)
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                sensor_range = self.sensor_ranges[column]

                # Cria uma máscara booleana: True para valores dentro do range
                mask = (df_clean[column] >= sensor_range['min']) & (df_clean[column] <= sensor_range['max'])
                # Mantém apenas as linhas que estão dentro do range ou são NaN (serão tratadas depois)
                df_clean = df_clean[mask | df_clean[column].isnull()]
        return df_clean, {}

# ==============================================================================
# 3. CLASSIFICAÇÃO DA QUALIDADE DO AR (BASEADA EM REGRAS)
# ==============================================================================
class AirQualityClassifier:
    """
    Classifica a qualidade do ar com base em regras e limites de saúde.
    - Calcula um "sub-índice" para cada poluente.
    - O pior sub-índice determina a classificação final da qualidade do ar.
    """
    def __init__(self, limits: Dict[str, float]):
        self.limits = limits  # Limites de qualidade do ar vindos da config
        # Categorias de qualidade do ar baseadas no valor do pior sub-índice
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
            # Fórmula do sub-índice: concentração medida / limite de qualidade
            sub_indices[f'SubIndice_{pollutant}'] = df_result[pollutant] / (limit_value if limit_value != 0 else 1.0)

        # O índice final é o valor máximo entre todos os sub-índices de uma medição
        df_result['Max_SubIndice'] = sub_indices.max(axis=1)
        # Aplica a categorização e avaliação de risco com base no índice final
        df_result['Qualidade_Ar_Calculada'] = df_result['Max_SubIndice'].apply(self._categorize)
        df_result['Risco_Saude'] = df_result['Max_SubIndice'].apply(self._assess_health_risk)
        return df_result

    def _categorize(self, idx):
        """Função auxiliar para mapear o valor do índice para uma categoria de qualidade."""
        if pd.isna(idx): return "Indeterminado"
        return next((cat for (min_v, max_v), cat in self.categories.items() if min_v <= idx < max_v), "Crítico")

    def _assess_health_risk(self, idx):
        """Função auxiliar para mapear o valor do índice para um nível de risco à saúde."""
        if pd.isna(idx): return "Indeterminado"
        if idx <= 0.5: return "Baixo"
        if idx <= 1.0: return "Moderado"
        if idx <= 1.5: return "Alto"
        return "Muito Alto"

# ==============================================================================
# 4. PREVISÃO DE SÉRIES TEMPORAIS (FORECASTING)
# ==============================================================================
class SimpleGasForecaster:
    """
    Realiza previsões de séries temporais para as concentrações futuras de gases.
    - Usa o modelo Exponential Smoothing, bom para dados com tendência e sazonalidade.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}  # Dicionário para armazenar um modelo treinado para cada gás
        self.target_gas_columns = config.get('target_gas_columns', [])
        self.min_data_for_train = config.get('min_data_for_train', 50)
        self.trained_on_index_freq = None

    def train(self, df_historical: pd.DataFrame) -> bool:
        """Treina um modelo para cada gás alvo usando dados históricos."""
        if isinstance(df_historical.index, pd.DatetimeIndex):
            # Tenta inferir a frequência dos dados (ex: horária, diária)
            self.trained_on_index_freq = pd.infer_freq(df_historical.index)

        for gas_col in self.target_gas_columns:
            if gas_col in df_historical.columns:
                series = df_historical[gas_col].dropna().astype(float)
                if len(series) >= self.min_data_for_train:
                    try:
                        # Se os dados forem horários, considera um ciclo diário (sazonalidade de 24 períodos)
                        seasonal_periods = 24 if self.trained_on_index_freq and 'H' in self.trained_on_index_freq else None
                        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
                        self.models[gas_col] = model
                    except Exception:
                        # Se o modelo falhar, usa o último valor como uma previsão simples (fallback)
                        self.models[gas_col] = series.iloc[-1]
                elif not series.empty:
                    self.models[gas_col] = series.iloc[-1]
        return bool(self.models) # Retorna True se algum modelo foi treinado

    def forecast(self, n_periods: int, last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Gera previsões para 'n_periods' no futuro."""
        if not self.models: return None

        forecast_data = {}
        last_ts = last_timestamp or datetime.now()
        freq = self.trained_on_index_freq or 'H' # Usa a frequência inferida ou 'H' (horária) como padrão
        # Cria os timestamps futuros para o DataFrame de previsão
        future_index = pd.date_range(start=last_ts + timedelta(hours=1), periods=n_periods, freq=freq)

        for gas_col, model in self.models.items():
            # Se o 'model' for um modelo treinado, chama .forecast(). Se for um valor (fallback), repete o valor.
            forecast_data[gas_col] = model.forecast(n_periods) if hasattr(model, 'forecast') else [model] * n_periods

        return pd.DataFrame(forecast_data, index=future_index)

# ==============================================================================
# 5. CLASSIFICAÇÃO COM MACHINE LEARNING (ÁRVORE DE DECISÃO)
# ==============================================================================
class DecisionTreePipeline:
    """
    Pipeline para treinar e usar um modelo de Árvore de Decisão.
    - Aprende com os dados históricos classificados pelas regras.
    - Pode classificar novas leituras e previsões com base no que aprendeu.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.model: Optional[DecisionTreeClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.class_names: List[str] = []
        self.is_trained = False

    def train(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> bool:
        """Treina o modelo de Árvore de Decisão."""
        self.feature_columns = feature_columns
        # Remove linhas com dados faltantes para garantir a qualidade do treino
        df_train = df.dropna(subset=[target_column] + feature_columns).copy()

        if len(df_train) < self.config.get('min_data_points_tree', 20):
            logger.warning(f"Dados insuficientes para treinar: {len(df_train)} amostras.")
            return False

        X = df_train[feature_columns]  # Variáveis de entrada (features)
        y = self.label_encoder.fit_transform(df_train[target_column])  # Variável alvo (target)
        self.class_names = list(self.label_encoder.classes_)

        # Divide os dados para treinar e para testar a performance do modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(self.class_names) > 1 else None)

        self.model = DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy')
        self.model.fit(X_train, y_train)  # Treina o modelo
        self.is_trained = True
        logger.info(f"Árvore de Decisão treinada. Acurácia no teste: {self.model.score(X_test, y_test):.4f}")
        # Se configurado, gera um gráfico da árvore
        if self.config.get('plot_tree_enabled', False):
             self._plot_decision_tree()
        return True

    def predict(self, df_features: pd.DataFrame) -> Optional[List[str]]:
        """Usa o modelo treinado para fazer novas previsões."""
        if not self.is_trained: return None
        # Preenche valores faltantes com a média para evitar erros na predição
        X_pred = df_features[self.feature_columns].fillna(df_features[self.feature_columns].mean())
        predictions_encoded = self.model.predict(X_pred)
        # Converte a predição numérica de volta para texto (ex: "Bom", "Ruim")
        return self.label_encoder.inverse_transform(predictions_encoded).tolist()

    def _plot_decision_tree(self):
        """Gera e salva uma imagem da Árvore de Decisão."""
        if not self.model: return
        try:
            plt.figure(figsize=(25, 15))
            plot_tree(self.model, feature_names=self.feature_columns, class_names=self.class_names, filled=True, rounded=True, fontsize=10)
            plt.savefig(self.config.get('plot_path', 'decision_tree.png'))
            plt.close()
            logger.info(f"Imagem da árvore de decisão salva em: {self.config.get('plot_path')}")
        except Exception as e:
            logger.error(f"Erro ao plotar árvore: {e}")

# ==============================================================================
# 6. CLASSE ORQUESTRADORA PRINCIPAL
# ==============================================================================

class SentinelaVerde:
    """
    Classe principal que orquestra todo o fluxo de trabalho.
    Agora ela também gerencia um buffer de dados recebidos via MQTT.
    """
    def __init__(self, config_path: str = None):
        # ... (inicialização dos componentes de análise inalterada) ...
        self.config_manager = ConfigManager() # Simplificado
        self.config = self.config_manager.DEFAULT_CONFIG # Simplificado
        self.data_buffer: Dict[str, Any] = {} # Buffer para agregar dados MQTT
        self.buffer_lock = threading.Lock() # Para garantir que o buffer seja acessado de forma segura

    def process_mqtt_message(self, topic: str, payload: str):
        """
        Processa uma mensagem recebida do broker MQTT.
        Agrega os dados no buffer até ter um conjunto completo.
        """
        logger.info(f"Mensagem recebida - Tópico: {topic}, Valor: {payload}")
        
        # Extrai o nome da métrica do tópico (ex: 'sentinela/temperatura' -> 'temperatura')
        metric_name = topic.split('/')[-1]
        
        with self.buffer_lock:
            # Armazena o valor no buffer
            try:
                self.data_buffer[metric_name] = float(payload)
            except ValueError:
                logger.warning(f"Não foi possível converter o payload '{payload}' para float.")
                return

            # Verifica se já temos todas as leituras necessárias
            required_keys = {"temperatura", "umidade", "gas"}
            if required_keys.issubset(self.data_buffer.keys()):
                logger.info("Buffer completo. Disparando análise...")
                
                # Mapeia os dados do buffer para o formato esperado pelo backend
                formatted_data = {
                    'Amonia_ppm': self.data_buffer.get('gas'), # Mapeia 'gas' para 'Amonia_ppm'
                    'Benzeno_ppm': None, # Deixado como None, conforme o pedido
                    'Alcool_ppm': None, # Deixado como None
                    'Dioxido_Carbono_ppm': None, # Deixado como None
                    'Temperatura_C': self.data_buffer.get('temperatura'),
                    'Umidade_Relativa_percent': self.data_buffer.get('umidade')
                }
                
                # Dispara a função de análise e salvamento
                self.append_new_reading_and_run_analysis(formatted_data)
                
                # Limpa o buffer para o próximo conjunto de leituras
                self.data_buffer.clear()

    def append_new_reading_and_run_analysis(self, data_dict: Dict):
        """Recebe os dados agregados, anexa ao CSV e executa um novo ciclo de análise."""
        try:
            data_dict['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df_new_row = pd.DataFrame([data_dict])
            input_file = self.config['files']['input_csv']
            
            # Garante que o arquivo CSV exista com cabeçalho
            if not Path(input_file).exists() or Path(input_file).stat().st_size == 0:
                df_new_row.to_csv(input_file, index=False)
            else:
                df_new_row.to_csv(input_file, mode='a', header=False, index=False)
            
            logger.info(f"Nova leitura adicionada a {input_file}")
            # self.run_analysis() # A análise completa seria executada aqui
        except Exception as e:
            logger.error(f"Falha ao anexar nova leitura: {e}", exc_info=True)
    def run_analysis(self):
        """Executa o ciclo completo de análise: carregar, classificar, treinar modelos e prever."""
        logger.info("--- INICIANDO CICLO DE ANÁLISE ---")
        if not self.load_data() or self.df_data is None or self.df_data.empty:
            logger.warning("Análise abortada: não foi possível carregar dados ou o arquivo está vazio.")
            self.latest_reading_data = None
            self.df_future_gases_forecast = None
            return

        # 1. Classificação baseada em regras
        self.df_classified_rules = self.rule_classifier.calculate_air_quality(self.df_data)
        if not self.df_classified_rules.empty:
            self.latest_reading_data = self.df_classified_rules.iloc[-1]
            self.latest_reading_rules_classification = self.latest_reading_data.get('Qualidade_Ar_Calculada')

        # 2. Treinamento e predição com a Árvore de Decisão (IA)
        if self.decision_tree_pipeline:
            dt_conf = self.config['decision_tree']
            if self.decision_tree_pipeline.train(self.df_classified_rules, dt_conf['feature_columns'], dt_conf['target_column']):
                if self.latest_reading_data is not None:
                    pred = self.decision_tree_pipeline.predict(pd.DataFrame([self.latest_reading_data]))
                    self.latest_reading_dt_classification = pred[0] if pred else "Falha na Predição IA"

        # 3. Treinamento e previsão de séries temporais
        if self.gas_forecaster and self.gas_forecaster.train(self.df_classified_rules):
            horizon = self.config['gas_forecasting']['prediction_horizon_hours']
            self.df_future_gases_forecast = self.gas_forecaster.forecast(horizon, self.last_timestamp_processed)

        logger.info("--- CICLO DE ANÁLISE CONCLUÍDO ---")

    def get_formatted_summary(self) -> str:
        """Retorna um resumo formatado dos resultados para ser exibido na interface do usuário."""
        if self.latest_reading_data is None:
            return "Nenhuma leitura de dados válida para exibir."

        summary = [f"ÚLTIMA LEITURA ({self.last_timestamp_processed.strftime('%d/%m/%Y %H:%M') if self.last_timestamp_processed else 'N/A'}):"]
        for sensor, value in self.latest_reading_data.items():
            if sensor in self.config['columns']['sensors']:
                summary.append(f"  - {sensor.replace('_ppm', '')}: {value:.2f} ppm")

        summary.append(f"\nQUALIDADE DO AR (REGRAS): {self.latest_reading_rules_classification or 'N/A'}")
        if self.latest_reading_dt_classification:
            summary.append(f"QUALIDADE DO AR (IA): {self.latest_reading_dt_classification}")

        if self.df_future_gases_forecast is not None:
            summary.append("\nPREVISÕES (Próximas 24h):")
            # Converte o DataFrame para string para uma exibição formatada
            summary.append(str(self.df_future_gases_forecast.head().to_string()))

        return "\n".join(summary)
    
# ==============================================================================
# 3. CLIENTE MQTT
# ==============================================================================
class MQTTClient:
    """Gerencia a conexão e a lógica de subscrição ao broker MQTT."""
    def __init__(self, broker_address: str, port: int, topics: List[str], sentinela_instance: SentinelaVerde):
        self.broker_address = broker_address
        self.port = port
        self.topics = topics
        self.sentinela = sentinela_instance
        self.client = mqtt.Client()

        # Define as funções de callback
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        """Callback executado quando o cliente se conecta ao broker."""
        if rc == 0:
            logger.info("Conectado ao Broker MQTT com sucesso!")
            # Assina todos os tópicos da lista
            for topic in self.topics:
                client.subscribe(topic)
                logger.info(f"Assinando o tópico: {topic}")
        else:
            logger.error(f"Falha ao conectar, código de retorno: {rc}\n")

    def on_message(self, client, userdata, msg):
        """Callback executado quando uma mensagem é recebida em um tópico assinado."""
        payload = msg.payload.decode()
        # Envia a mensagem para a instância do SentinelaVerde processar
        self.sentinela.process_mqtt_message(msg.topic, payload)

    def start(self):
        """Inicia a conexão e o loop de escuta do cliente MQTT."""
        try:
            self.client.connect(self.broker_address, self.port, 60)
            # Inicia o loop em uma thread separada para não bloquear a aplicação principal
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Não foi possível conectar ao broker MQTT: {e}")
