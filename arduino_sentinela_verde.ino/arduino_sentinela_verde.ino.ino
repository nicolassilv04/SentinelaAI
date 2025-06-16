/**
 * @file arduino_sentinela_verde.ino
 * @author Nicolas
 * @brief Código para ESP32 do projeto Sentinela Verde.
 * @version 3.0 (Robusta)
 * * @details
 * Este código realiza as seguintes tarefas:
 * 1. Conecta-se a uma rede Wi-Fi.
 * 2. Conecta-se a um broker MQTT público.
 * 3. Lê os dados dos sensores DHT11 (temperatura e umidade) e MQ-135 (qualidade do ar).
 * 4. Junta todos os dados em UMA ÚNICA MENSAGEM de texto, separada por vírgulas.
 * Formato: "temperatura,umidade,concentracao_ppm"
 * 5. Publica esta mensagem única em um TÓPICO MQTT único a cada minuto.
 * Este método é mais confiável do que enviar múltiplas mensagens separadas.
 * * Bibliotecas necessárias (Instalar pela Library Manager da Arduino IDE):
 * 1. "DHT sensor library" by Adafruit
 * 2. "Adafruit Unified Sensor" by Adafruit
 * 3. "PubSubClient" by Nick O'Leary
 */

// ==============================================================================
// 1. INCLUSÃO DE BIBLIOTECAS
// ==============================================================================
#include <WiFi.h>          // Biblioteca para funcionalidades Wi-Fi do ESP32.
#include <PubSubClient.h>  // Biblioteca para comunicação via protocolo MQTT.
#include <DHT.h>           // Biblioteca para o sensor de temperatura e umidade DHT.

// ==============================================================================
// 2. CONFIGURAÇÕES - EDITE ESTAS LINHAS CONFORME NECESSÁRIO
// ==============================================================================
const char* ssid = "Arduino";        // <-- COLOQUE AQUI O NOME DA SUA REDE WI-FI
const char* password = "ArduinoTeste";  // <-- COLOQUE AQUI A SENHA DA SUA REDE WI-FI
const char* mqtt_server = "test.mosquitto.org"; // Broker MQTT público para testes.
const char* mqtt_topic = "sentinela/dados_csv";   // Tópico ÚNICO para enviar todos os dados.

// --- Pinos dos Sensores ---
#define DHT_PIN 13          // Pino digital onde o sensor DHT11 está conectado.
#define DHT_TYPE DHT11      // Define o tipo do sensor DHT (pode ser DHT22, etc.).
#define MQ135_ANALOG_PIN 4  // Pino analógico onde o sensor MQ-135 está conectado.

// ==============================================================================
// 3. INICIALIZAÇÃO DE OBJETOS E VARIÁVEIS GLOBAIS
// ==============================================================================
WiFiClient espClient; // Cria um cliente Wi-Fi para a conexão de rede.
PubSubClient client(espClient); // Cria um cliente MQTT usando o cliente Wi-Fi.
DHT dht(DHT_PIN, DHT_TYPE);     // Inicializa o objeto do sensor DHT.

unsigned long lastMsg = 0; // Variável para armazenar o tempo da última mensagem enviada.
                           // Usada para criar um intervalo não-bloqueante (sem usar delay()).

// ==============================================================================
// 4. FUNÇÃO SETUP - EXECUTADA UMA ÚNICA VEZ QUANDO O ESP32 LIGA
// ==============================================================================
void setup() {
  // Inicia a comunicação serial a 115200 bauds para depuração no Serial Monitor.
  Serial.begin(115200);
  
  // Chama a função para configurar e conectar ao Wi-Fi.
  setup_wifi();
  
  // Configura o cliente MQTT com o endereço e porta do broker.
  client.setServer(mqtt_server, 1883);
  
  // Inicia o sensor DHT.
  dht.begin();
}

// ==============================================================================
// 5. FUNÇÃO LOOP - EXECUTADA REPETIDAMENTE EM CICLO
// ==============================================================================
void loop() {
  // Garante que o cliente MQTT esteja sempre conectado. Se não estiver, chama a função de reconexão.
  if (!client.connected()) {
    reconnect();
  }
  // Mantém o cliente MQTT processando mensagens de entrada/saída. Deve ser chamado regularmente.
  client.loop();

  // Lógica de temporização não-bloqueante.
  unsigned long now = millis(); // Pega o tempo atual em milissegundos.
  if (now - lastMsg > 2000) {  // Verifica se já se passou 1 minuto (60000 ms).
    lastMsg = now; // Atualiza o tempo da última mensagem para a contagem do próximo ciclo.

    // --- Leitura dos Sensores ---
    float temp = dht.readTemperature();
    float humid = dht.readHumidity();
    int gas_analog = analogRead(MQ135_ANALOG_PIN); // Lê o valor analógico bruto do MQ-135 (0-4095).

    // --- Validação da Leitura ---
    // Verifica se a leitura do DHT falhou (retorna 'Not a Number').
    if (isnan(temp) || isnan(humid)) {
      Serial.println("Falha ao ler do sensor DHT! Verifique a conexão.");
      return; // Pula o resto do loop e tenta novamente no próximo ciclo.
    }
    
    // Converte o valor analógico para uma estimativa de PPM (Partes Por Milhão).
    // Esta é uma conversão SIMPLES e LINEAR. Para precisão, uma calibração é necessária.
    float concentracao_geral_ppm = map(gas_analog, 0, 4095, 10, 1000);

    // --- Criação do Payload (Carga de Dados) ---
    // Concatena todos os valores em uma única String, separados por vírgula.
    // Exemplo de resultado: "25.50,60.80,450.00"
    String payload = String(temp) + "," + String(humid) + "," + String(concentracao_geral_ppm);
    
    // --- Publicação no Tópico MQTT ---
    // Publica a String 'payload' no tópico definido. O '.c_str()' converte a String do Arduino para o formato C.
    if (client.publish(mqtt_topic, payload.c_str())) {
        Serial.println("---------------------------------");
        Serial.print("Mensagem enviada para o topico sentinela/dados_csv: ");
        Serial.println(payload);
        Serial.println("---------------------------------");
    } else {
        Serial.println("Falha ao publicar mensagem MQTT.");
    }
  }
}

// ==============================================================================
// 6. FUNÇÕES AUXILIARES
// ==============================================================================

/**
 * @brief Configura e conecta o ESP32 à rede Wi-Fi.
 */
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Conectando a ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi conectado!");
  Serial.print("Endereco IP: ");
  Serial.println(WiFi.localIP());
}

/**
 * @brief Tenta se reconectar ao broker MQTT em caso de desconexão.
 */
void reconnect() {
  // Entra em loop até que a reconexão seja bem-sucedida.
  while (!client.connected()) {
    Serial.print("Tentando conexao MQTT...");
    // Cria um ID de cliente único para evitar conflitos no broker.
    String clientId = "ESP32Client-SentinelaVerde-";
    clientId += String(random(0xffff), HEX);
    
    // Tenta conectar.
    if (client.connect(clientId.c_str())) {
      Serial.println("Conectado!");
      // Se houver necessidade de receber dados, a subscrição a tópicos seria feita aqui.
      // Ex: client.subscribe("meu/topico/de/comando");
    } else {
      Serial.print("falhou, rc=");
      Serial.print(client.state());
      Serial.println(" tentando novamente em 5 segundos");
      // Espera 5 segundos antes de tentar novamente.
      delay(5000);
    }
  }
}
