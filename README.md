Sentinela Verde - Monitoramento Preditivo da Qualidade do Ar
O Sentinela Verde é um sistema completo para monitoramento, análise e previsão da qualidade do ar em tempo real, utilizando Python, Machine Learning e um dashboard interativo. Ele transforma dados brutos de sensores de baixo custo em insights acionáveis para a vigilância proativa da saúde ambiental.

Demonstração
(Interface principal do sistema, exibindo dados em tempo real e previsões futuras)

Principais Funcionalidades
Monitoramento em Tempo Real: Acompanhe as medições de gases (Amônia, Benzeno, Álcool, CO₂), temperatura e umidade com atualização automática.

Classificação Híbrida:

Baseada em Regras: Utiliza limites de saúde (inspirados na NR-15) para uma classificação imediata e transparente da qualidade do ar.

Baseada em IA: Emprega um modelo de Árvore de Decisão que aprende os padrões dos dados para fornecer uma classificação inteligente.

Previsão do Futuro (Forecasting): Utiliza modelos de séries temporais (Exponential Smoothing) para prever as concentrações de gases nas próximas 24 horas, permitindo ações preventivas.

Dashboard Interativo: Uma interface gráfica construída com Flet para visualizar dados, gráficos e relatórios de forma clara e intuitiva.

Sistema Configurável: Todos os parâmetros importantes, como limites de gases e caminhos de arquivos, podem ser ajustados no arquivo config.yaml sem a necessidade de alterar o código.

Como Funciona (Arquitetura)
O sistema é dividido em três camadas principais:

Coleta de Dados (Hardware):

Um microcontrolador Arduino com um sensor de qualidade do ar MQ-135 e um sensor de ambiente DHT11/22 coleta os dados brutos.

Os dados são enviados via serial para um computador e armazenados em um arquivo meus_dados_arduino_historico.csv, que funciona como o banco de dados histórico do projeto.

Processamento e Inteligência (Backend - backend.py):

Este é o cérebro do sistema, escrito em Python.

Ele carrega os dados do CSV, valida as leituras para garantir que estejam dentro dos limites operacionais do sensor, e aplica as técnicas de classificação e previsão.

Orquestra os modelos de Machine Learning e estatística.

Visualização (Frontend - frontend4.py):

A interface gráfica do usuário, construída com o framework Flet.

Comunica-se com o backend para solicitar análises e exibe os resultados de forma amigável, com gráficos e painéis que são atualizados em tempo real.

Tecnologias Utilizadas
Backend: Python 3

Frontend: Flet

Análise de Dados: Pandas, NumPy

Machine Learning: Scikit-learn (para a Árvore de Decisão)

Previsão de Séries Temporais: Statsmodels

Visualização de Dados (Gráficos): Matplotlib

Configuração: PyYAML

Estrutura do Projeto
/
├── backend.py                  # Script principal da lógica e análise de dados.
├── frontend4.py                # Script da interface gráfica do usuário (dashboard).
├── config.yaml                 # Arquivo de configuração para todos os parâmetros.
├── requirements.txt            # Lista de dependências Python para instalação.
├── meus_dados_arduino_historico.csv  # Banco de dados histórico com as leituras.
├── decision_tree_air_quality.png # Imagem da árvore de decisão gerada pela IA.
└── sentinela_verde.log         # Arquivo de log para registro de eventos e erros.


Instalação e Execução
Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

1. Pré-requisitos
Python 3.9 ou superior.

2. Clone o Repositório
git clone https://github.com/seu-usuario/sentinela-verde.git
cd sentinela-verde


3. Crie um Ambiente Virtual
É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


4. Instale as Dependências
Crie um arquivo chamado requirements.txt com o seguinte conteúdo:

pandas
numpy
PyYAML
scikit-learn
matplotlib
statsmodels
flet
Pillow


Em seguida, instale todas as bibliotecas de uma vez com o pip:

pip install -r requirements.txt


5. Execute a Aplicação
Para iniciar o dashboard, execute o script do frontend:

python frontend4.py


A janela do Sentinela Verde deverá abrir, carregando os dados do arquivo meus_dados_arduino_historico.csv e exibindo a primeira análise.

Configuração
Você pode customizar o comportamento do sistema editando o arquivo config.yaml. Algumas das principais configurações que você pode ajustar são:

air_quality_limits: Altere os limites de ppm para cada gás para ajustar o gatilho da classificação "Ruim".

sensor_ranges: Modifique os valores mínimos e máximos para corresponder às especificações do seu sensor.

prediction_horizon_hours: Aumente ou diminua o número de horas que o sistema deve prever.

Próximos Passos e Melhorias Futuras
[ ] Migração para Banco de Dados: Substituir o arquivo CSV por um banco de dados mais robusto como SQLite, PostgreSQL ou Firebase para melhor escalabilidade.

[ ] Sistema de Alertas: Implementar notificações por e-mail ou Telegram quando a previsão indicar um risco iminente.

[ ] Deploy na Nuvem: Empacotar a aplicação com Docker e fazer o deploy em um serviço de nuvem para acesso remoto.

[ ] Expandir Sensores: Adicionar suporte a outros sensores, como o de material particulado (PM2.5).
