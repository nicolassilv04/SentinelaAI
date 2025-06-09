# frontend4.py (versão com atualização em tempo real e comentários detalhados)

# --- Importações de Bibliotecas Essenciais ---
import flet as ft  # Biblioteca principal para criar a interface gráfica
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para rodar sem interface gráfica (necessário para Flet)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import io  # Para manipular dados em memória (streams de bytes)
import base64  # Para codificar imagens em texto e exibi-las no Flet
import backend  # Importa todo o nosso código de lógica de negócio
import time  # Para a pausa no loop de atualização automática
import threading  # Para executar a atualização automática em segundo plano sem travar a UI

# --- Verificação de Dependências Opcionais ---
try:
    from PIL import Image as PILImage  # Pillow é necessário para exibir a imagem da árvore de decisão
    PIL_DISPONIVEL = True
except ImportError:
    PIL_DISPONIVEL = False
    PILImage = None

# --- Instância do Backend e Configurações Globais ---
# Cria a instância principal do nosso sistema de backend.
# O frontend irá interagir com este objeto 'sentinela' para obter todos os dados e análises.
sentinela = backend.SentinelaVerde()
nomes_gases_configurados = sentinela.config.get('gas_forecasting', {}).get('target_gas_columns', [])

# --- Referências Globais a Controles Flet ---
# Criar referências globais para os controles da UI permite que eles sejam atualizados
# de qualquer lugar do código, especialmente nas funções de callback.
mapa_textos_valores_gases = {} # Armazena os controles de texto para os valores dos gases
texto_umidade = ft.Text("...")
texto_temperatura = ft.Text("...")
icone_qualidade_ar = ft.Icon(ft.Icons.HELP_OUTLINE)
texto_qualidade_ar = ft.Text("...")
container_qualidade_ar = ft.Container() # Container que muda de cor
controle_imagem_plot = ft.Image(fit=ft.ImageFit.CONTAIN, expand=True) # Onde o gráfico será exibido
area_texto_relatorio = ft.Text("", expand=True, selectable=True, font_family="Consolas")
area_imagem_relatorio = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, expand=True)
indicador_carregamento = ft.ProgressRing(visible=False, width=24, height=24) # Animação de "carregando"
switch_atualizacao_automatica = ft.Switch(label="Atualização em tempo real", value=False)

# --- Controle da Thread de Atualização ---
# Evento para sinalizar à thread que ela deve parar (quando o app fechar)
parar_thread_atualizacao = threading.Event()


# ==============================================================================
# FUNÇÕES DE LÓGICA E ATUALIZAÇÃO DA UI
# ==============================================================================

def gerar_imagem_grafico_base64(page_theme_mode=ft.ThemeMode.LIGHT) -> str:
    """
    Cria um gráfico de previsão de gases usando Matplotlib, o salva em memória
    e o retorna como uma string base64 para ser exibido no Flet.
    """
    figura_matplotlib = Figure(figsize=(7, 3.8), dpi=100)
    eixos = figura_matplotlib.add_subplot(111)

    # Adapta as cores do gráfico ao tema da página (claro ou escuro)
    cor_texto = 'white' if page_theme_mode == ft.ThemeMode.DARK else 'black'
    figura_matplotlib.set_facecolor('none') # Fundo transparente
    eixos.set_facecolor('none')

    tem_dados = False
    if sentinela.df_future_gases_forecast is not None:
        for gas in nomes_gases_configurados:
            if gas in sentinela.df_future_gases_forecast.columns:
                tem_dados = True
                eixos.plot(sentinela.df_future_gases_forecast.index, sentinela.df_future_gases_forecast[gas], label=gas.replace('_ppm', ''))

    if tem_dados:
        eixos.legend(prop={'size': 8}, labelcolor=cor_texto)
    else:
        eixos.text(0.5, 0.5, "Sem dados de previsão", ha='center', va='center', color=cor_texto)

    # Estiliza os eixos e as bordas
    eixos.tick_params(axis='x', colors=cor_texto, rotation=30, labelsize=8)
    eixos.tick_params(axis='y', colors=cor_texto, labelsize=8)
    for spine in eixos.spines.values():
        spine.set_edgecolor(cor_texto)

    figura_matplotlib.tight_layout()
    # Salva o gráfico em um buffer de bytes em memória
    buf = io.BytesIO()
    figura_matplotlib.savefig(buf, format="png", transparent=True)
    # Codifica os bytes da imagem em base64 e retorna como string
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def atualizar_elementos_ui(page: ft.Page):
    """
    Função central que busca os dados mais recentes do backend e atualiza todos
    os elementos visuais na interface do usuário.
    """
    indicador_carregamento.visible = True
    page.update() # Atualiza a UI para mostrar o indicador de carregamento

    # 1. Roda a análise completa no backend. Isso carrega os dados mais recentes do CSV e reprocessa tudo.
    sentinela.run_analysis()

    # 2. Atualiza os valores dos cards de gases se houver dados
    if sentinela.latest_reading_data is not None:
        for nome_gas in nomes_gases_configurados:
            if nome_gas in mapa_textos_valores_gases:
                valor = sentinela.latest_reading_data.get(nome_gas, 0)
                mapa_textos_valores_gases[nome_gas].value = f"{valor:.2f} ppm"

        # 3. Atualiza os dados de ambiente (temperatura e umidade)
        texto_umidade.value = f"Umidade: {sentinela.latest_reading_data.get('Umidade_Relativa_percent', 0):.1f}%"
        texto_temperatura.value = f"Temperatura: {sentinela.latest_reading_data.get('Temperatura_C', 0):.1f}°C"

        # 4. Atualiza o painel de qualidade do ar (texto e cor)
        qualidade = sentinela.latest_reading_dt_classification or sentinela.latest_reading_rules_classification or "Indisponível"
        texto_qualidade_ar.value = qualidade
        mapa_feedback = {'Excelente': ft.colors.GREEN, 'Bom': ft.colors.LIGHT_GREEN, 'Regular': ft.colors.AMBER, 'Ruim': ft.colors.ORANGE, 'Muito Ruim': ft.colors.RED, 'Crítico': ft.colors.PURPLE}
        container_qualidade_ar.bgcolor = mapa_feedback.get(qualidade, ft.colors.BLUE_GREY)

    # 5. Atualiza o gráfico de previsão
    controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(page.theme_mode)

    indicador_carregamento.visible = False
    page.update() # Atualiza a UI com todos os novos dados

def exibir_relatorio(page: ft.Page, tipo: str):
    """Exibe diferentes tipos de relatórios na aba 'Relatórios'."""
    area_texto_relatorio.visible = False
    area_imagem_relatorio.visible = False

    if tipo == "resumo":
        area_texto_relatorio.value = sentinela.get_formatted_summary()
        area_texto_relatorio.visible = True
    elif tipo == "arvore" and PIL_DISPONIVEL:
        try:
            caminho_imagem = sentinela.config['decision_tree']['plot_path']
            with open(caminho_imagem, "rb") as f:
                area_imagem_relatorio.src_base64 = base64.b64encode(f.read()).decode('utf-8')
            area_imagem_relatorio.visible = True
        except Exception as e:
            area_texto_relatorio.value = f"Erro ao carregar imagem da árvore: {e}"
            area_texto_relatorio.visible = True
    elif tipo == "log":
        try:
            with open('sentinela_verde.log', 'r', encoding='utf-8') as f:
                area_texto_relatorio.value = f.read()
            area_texto_relatorio.visible = True
        except Exception as e:
             area_texto_relatorio.value = f"Erro ao ler log: {e}"
             area_texto_relatorio.visible = True

    page.update()

def realtime_update_loop(page: ft.Page):
    """
    Loop que executa em uma thread separada para atualizações automáticas.
    Verifica a cada 10 segundos se a atualização automática está ligada.
    """
    while not parar_thread_atualizacao.is_set():
        if switch_atualizacao_automatica.value:
            try:
                print("Atualização automática executada.")
                atualizar_elementos_ui(page)
            except Exception as e:
                print(f"Erro na atualização automática: {e}")
        time.sleep(10) # Pausa de 10 segundos

# ==============================================================================
# FUNÇÃO PRINCIPAL DA APLICAÇÃO FLET
# ==============================================================================
def main(page: ft.Page):
    """Função principal que constrói e configura a interface gráfica."""
    page.title = "Sentinela Verde Ambiental"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.fonts = {"Consolas": "Consolas, 'Courier New', monospace"} # Fonte para relatórios

    # --- Inicialização da Thread de Background ---
    # Cria e inicia a thread que cuidará das atualizações automáticas
    update_thread = threading.Thread(target=realtime_update_loop, args=(page,), daemon=True)
    update_thread.start()

    # Garante que a thread pare quando o usuário fechar a janela
    def on_disconnect(e):
        parar_thread_atualizacao.set()
    page.on_disconnect = on_disconnect

    # --- Construção da Interface Gráfica (UI) ---

    # Cards de Gases (criados dinamicamente a partir da config)
    cards_gases = []
    for nome_gas in nomes_gases_configurados:
        texto_valor = ft.Text("-- ppm", weight=ft.FontWeight.BOLD, size=20)
        mapa_textos_valores_gases[nome_gas] = texto_valor
        cards_gases.append(
            ft.Card(content=ft.Container(
                content=ft.Column([ft.Text(nome_gas.replace('_ppm', ''), weight=ft.FontWeight.BOLD), texto_valor]),
                padding=15
            ), expand=True)
        )

    # Painel de Qualidade do Ar
    container_qualidade_ar.content = ft.Column([
        ft.Row([icone_qualidade_ar, ft.Text("Qualidade do Ar")]),
        texto_qualidade_ar
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    container_qualidade_ar.padding = 15
    container_qualidade_ar.border_radius = 10

    # Botão de atualização manual
    botao_atualizar = ft.FilledButton("Atualizar Dados", icon=ft.icons.REFRESH, on_click=lambda e: atualizar_elementos_ui(page))

    # --- Layout das Abas ---

    # Conteúdo da Aba de Monitoramento
    conteudo_aba_monitoramento = ft.Column([
        ft.Row(cards_gases),
        ft.Row([
            ft.Card(content=ft.Container(controle_imagem_plot, padding=10), expand=3),
            ft.Column([
                ft.Card(content=ft.Container(ft.Column([texto_umidade, texto_temperatura]), padding=15)),
                ft.Card(content=container_qualidade_ar)
            ], expand=1)
        ]),
        ft.Row([botao_atualizar, switch_atualizacao_automatica, indicador_carregamento], alignment=ft.MainAxisAlignment.CENTER)
    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)

    # Conteúdo da Aba de Relatórios
    conteudo_aba_relatorios = ft.Column([
        ft.Row([
            ft.ElevatedButton("Resumo", on_click=lambda e: exibir_relatorio(page, "resumo")),
            ft.ElevatedButton("Árvore de Decisão", on_click=lambda e: exibir_relatorio(page, "arvore"), disabled=not PIL_DISPONIVEL),
            ft.ElevatedButton("Log", on_click=lambda e: exibir_relatorio(page, "log")),
        ]),
        ft.Container(
            content=ft.Column([area_texto_relatorio, area_imagem_relatorio], scroll=ft.ScrollMode.ADAPTIVE, expand=True),
            border=ft.border.all(1, ft.colors.outline),
            border_radius=10,
            padding=10,
            expand=True
        )
    ], expand=True)

    # Controle de Abas principal
    abas = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Monitoramento", content=conteudo_aba_monitoramento),
            ft.Tab(text="Relatórios", content=conteudo_aba_relatorios),
        ],
        expand=True,
    )

    page.add(abas)
    atualizar_elementos_ui(page) # Faz uma carga inicial dos dados ao abrir o app

# Ponto de entrada da aplicação Flet
ft.app(target=main)
