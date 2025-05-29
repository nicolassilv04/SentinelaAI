# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys
from sklearn.model_selection import train_test_split

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

# TAMANHO DATASET
num_linhas = 580 * 2 # Ex: aprox 48 dias com dados de hora em hora

np.random.seed(42)

# Simular alguma variação diária (exemplo simples com senoide para CO2)
time_factor_co2 = np.sin(np.linspace(0, 8 * np.pi, num_linhas)) * 100 # Variação para CO2
time_factor_temp = np.sin(np.linspace(0, 8 * np.pi, num_linhas)) * 5 # Variação para Temperatura

data = {
    'Amonia_ppm': np.random.uniform(0.0025, 0.057, num_linhas).round(4), # Ajustei o range e arredondamento
    'Benzeno_ppm': np.random.uniform(0.001, 0.013, num_linhas).round(4),
    'Alcool_ppm': np.random.uniform(0.025, 0.47, num_linhas).round(4),
    'Dioxido_Carbono_ppm': np.random.randint(380, 600, num_linhas) + time_factor_co2.astype(int),
    'Temperatura_C': np.random.randint(15, 35, num_linhas) + time_factor_temp.astype(int),
    'Umidade_Relativa_percent': np.random.randint(30, 80, num_linhas)
}
df = pd.DataFrame(data)
df['Dioxido_Carbono_ppm'] = df['Dioxido_Carbono_ppm'].clip(lower=350) # Garantir que não fique irrealisticamente baixo

# DEFININDO QUALIDADE DO AR
# Limites de referência (EXEMPLOS - PESQUISE VALORES OFICIAIS OU DEFINA OS SEUS CRITÉRIOS)
# Unidades devem ser consistentes com os dados gerados.
limits = {
    'Amonia_ppm': 0.2,  # Ex: 200 ppb = 0.2 ppm (verificar padrões)
    'Benzeno_ppm': 0.005, # Ex: 5 ppb (verificar padrões)
    'Alcool_ppm': 1.0,    # Ex: 1 ppm (álcoois podem variar muito, este é apenas um placeholder)
    'Dioxido_Carbono_ppm': 1000 # Limite para qualidade "boa" em ambientes internos, por exemplo
}

concentrations_to_evaluate = ['Amonia_ppm', 'Benzeno_ppm', 'Alcool_ppm', 'Dioxido_Carbono_ppm'] # Poluentes para o índice

# Cálculo dos sub-índices
# Garante que apenas as colunas presentes em 'limits' e 'df' sejam usadas
valid_pollutants = [p for p in concentrations_to_evaluate if p in df.columns and p in limits]

sub_indices_df = pd.DataFrame()
for pollutant in valid_pollutants:
    sub_indices_df[pollutant] = df[pollutant] / limits[pollutant]

df['Soma_Subindices'] = sub_indices_df.sum(axis=1)

# Ajuste estes thresholds com base na sua escala de Soma_Subindices!
# Exemplo de thresholds (precisam ser validados/ajustados):
# Se a soma dos sub-indices (onde 1.0 por poluente é o limite) resultar em valores
# tipicamente entre 0 e 4 (para 4 poluentes), os thresholds seriam algo como:
qualidade = []
threshold_bom = 1.0      # Soma total <= 1.0 (todos bem abaixo do limite)
threshold_regular = 2.0  # Soma total <= 2.0
threshold_poluido = 3.0  # Soma total <= 3.0

for valor_soma in df['Soma_Subindices']:
    if valor_soma <= threshold_bom:
        qualidade.append("Bom")
    elif valor_soma <= threshold_regular:
        qualidade.append("Regular")
    elif valor_soma <= threshold_poluido:
        qualidade.append("Poluido")
    else:
        qualidade.append("Critico")

df['Qualidade_Ar'] = qualidade

# Definir Timestamp como índice para facilitar operações de séries temporais
#df.set_index('Timestamp', inplace=True)

# CONVERTENDO DATAFRAME PARA ARQUIVO .CSV
df.to_csv('sentinela_aprimorado.csv')

# EXIBINDO AS PRIMEIRAS LINHAS DO DATAFRAME
print("DataFrame Aprimorado:")
print(df.head())
print("\nContagem de Categorias de Qualidade do Ar:")
print(df['Qualidade_Ar'].value_counts())

