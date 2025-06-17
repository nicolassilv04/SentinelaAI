# 🌳 Sentinela Verde
O Sentinela Verde é um sistema completo de monitoramento da qualidade do ar que utiliza sensores de hardware (ESP32), comunicação via MQTT e inteligência artificial para coletar, processar, analisar e visualizar dados ambientais em tempo real.<br>

O sistema é composto por um backend robusto em Python que gerencia os dados e os modelos de Machine Learning, e uma interface gráfica de desktop moderna e reativa construída com Flet, permitindo uma análise detalhada e preditiva das condições do ar.<br>


### ✨ Funcionalidades Principais

- Coleta de Dados em Tempo Real: Recebe dados de temperatura, umidade e concentração de gases enviados por um microcontrolador (como um ESP32) via protocolo MQTT.

- Simulador de Dados para Testes: Inclui um gerador de dados para uso caseiro, perfeito para testar a aplicação sem a necessidade do hardware físico (Arduino/ESP32).

- Integração com API Externa: Enriquece os dados locais com informações de poluentes (PM2.5, PM10) da API pública World Air Quality Index (WAQI).

- Classificação com IA: Utiliza um modelo de Árvore de Decisão para classificar a qualidade do ar local com base nos dados combinados dos sensores.

- Previsão Futura: Emprega um modelo de série temporal (Holt-Winters) para prever os níveis de poluentes nas próximas 24 horas.

- Dashboard Interativo: Uma interface gráfica construída com Flet exibe os dados atuais, a classificação da IA, a previsão futura e gráficos de tendência.

- Relatórios e Recomendações: Gera resumos de dados e oferece recomendações de saúde com base na qualidade do ar detectada.

- Altamente Configurável: Todas as configurações, como chaves de API, tópicos MQTT e parâmetros de modelo, são gerenciadas em um único arquivo ````config.yaml.````

- Logging Completo: Registra todos os eventos importantes, desde a conexão MQTT até o treinamento dos modelos, em um arquivo de log para fácil depuração.


### 🏛️ Estrutura do Projeto
O projeto é modularizado para separar as responsabilidades, facilitando a manutenção e a escalabilidade.

````/
├── main_app.py               # Ponto de entrada principal da aplicação Flet.
├── frontend.py               # Define toda a interface gráfica e seus componentes.
├── backend.py                # Contém a lógica de negócio, processamento e os modelos de IA.
├── gerador_de_dados.py       # Script para simular o envio de dados do sensor via MQTT.
├── api_client.py             # Módulo para se comunicar com a API externa do WAQI.
├── config.yaml               # Arquivo central para todas as configurações do projeto.
└── sentinela_arduino.txt     # (Referência) Código para o microcontrolador ESP32.
````

### 🛠️ Como Executar o Projeto
Siga os passos abaixo para configurar e executar o Sentinela Verde em seu ambiente local.

#### 1. Pré-requisitos
Python 3.8+

Hardware (Opcional):

ESP32

Sensor de Temperatura e Umidade (DHT11)

Sensor de Qualidade do Ar (MQ-135)

Arduino IDE (se for usar o hardware) com as seguintes bibliotecas:

````DHT sensor library```` by Adafruit

````Adafruit Unified Sensor```` by Adafruit

````PubSubClient```` by Nick O'Leary

#### 2. Instalação
a. Clone o repositório:
````
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
````

b. Crie um ambiente virtual e instale as dependências:
````
# Crie um ambiente virtual (recomendado)
python -m venv venv
# Ative o ambiente (Windows)
.\venv\Scripts\activate
# Ative o ambiente (macOS/Linux)
source venv/bin/activate

# Instale as bibliotecas necessárias
pip install flet pandas paho-mqtt PyYAML scikit-learn statsmodels matplotlib requests
````
#### 3. Configuração
Antes de executar, você precisa configurar o arquivo ````config.yaml````.

a. Obtenha uma chave de API do WAQI:

A aplicação utiliza a API do WAQI para buscar dados de PM2.5 e PM10.

Obtenha uma chave de API gratuita em: https://aqicn.org/data-platform/token/

b. Edite o ````config.yaml````:
````
api:
  # Cole a sua chave de API obtida no passo anterior
  token: "SUA_CHAVE_API_AQUI" 
  city: "rio claro"
````
Altere o campo ````token```` com a chave que você obteve.

Você pode alterar a ````city```` e outros parâmetros, como o tópico MQTT (````topic````), se desejar.

#### 4. Escolha uma Fonte de Dados
Você pode executar a aplicação usando o hardware real (ESP32) ou o simulador de dados.

- Opção A: Usando o Hardware (ESP32)
Abra o código de ````arduino_sentinela_verde.ino```` na ````Arduino IDE.

Insira as credenciais da sua rede Wi-Fi nos campos ````ssid```` e ````password````.

Garanta que o ````mqtt_topic```` no código Arduino seja o mesmo definido em seu ````config.yaml````.

Carregue o código para o seu ESP32. Ele começará a enviar dados para o broker MQTT.<br></br>


- Opção B: Usando o Gerador de Dados (Sem Hardware)
Este script é ideal para uso caseiro ou para desenvolvimento, caso você não possua o hardware Arduino/ESP32.

Abra um novo terminal na pasta do projeto (com o ambiente virtual ativado).

Execute o ````gerador_de_dados.py````:
````
python gerador_de_dados.py
````
O terminal começará a exibir os dados simulados que estão sendo enviados.

Nota Importante: Você pode facilmente alterar o range de geração dos dados para simular diferentes cenários. Para isso, basta editar os valores dentro das funções ````random.uniform()```` no arquivo ````gerador_de_dados.py````.

#### 5. Execute a Aplicação Principal
Com a fonte de dados (real ou simulada) em execução, abra outro terminal (com o ambiente virtual ativado) e inicie a aplicação Flet.
````
python main_app.py
````
A interface gráfica do Sentinela Verde será iniciada, pronta para receber, processar e exibir os dados.

