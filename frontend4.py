# frontend4.py (versão corrigida com atualização em tempo real)

import flet as ft
import matplotlib
matplotlib.use('Agg') # Essencial para ambientes sem GUI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import io
import base64
import backend
import time
import threading

# Tenta importar Pillow
try:
    from PIL import Image as PILImage
    PIL_DISPONIVEL = True
except ImportError:
    PIL_DISPONIVEL = False
    PILImage = None

# --- Instância do Backend e Configurações Globais ---
sentinela = backend.SentinelaVerde()
nomes_gases_configurados = sentinela.config.get('gas_forecasting', {}).get('target_gas_columns', [])

# --- Referências Globais a Controles Flet ---
mapa_textos_valores_gases = {}
mapa_alertas_icones_gases = {}
texto_umidade = ft.Text("...")
texto_temperatura = ft.Text("...")
icone_qualidade_ar = ft.Icon(ft.Icons.HELP_OUTLINE)
texto_qualidade_ar = ft.Text("...")
container_qualidade_ar = ft.Container()
controle_imagem_plot = ft.Image(fit=ft.ImageFit.CONTAIN, expand=True)
area_texto_relatorio = ft.Text("", expand=True, selectable=True, font_family="Consolas")
area_imagem_relatorio = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, expand=True)
indicador_carregamento = ft.ProgressRing(visible=False, width=24, height=24)
switch_atualizacao_automatica = ft.Switch(label="Atualização em tempo real", value=False)

# Variáveis de controle para o loop de atualização
parar_thread_atualizacao = threading.Event()

# --- Funções de Lógica e UI ---

def gerar_imagem_grafico_base64(page_theme_mode=ft.ThemeMode.LIGHT):
    figura_matplotlib = Figure(figsize=(7, 3.8), dpi=100)
    eixos = figura_matplotlib.add_subplot(111)
    
    # Adaptação ao tema
    cor_texto = 'white' if page_theme_mode == ft.ThemeMode.DARK else 'black'
    figura_matplotlib.set_facecolor('none')
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

    eixos.tick_params(axis='x', colors=cor_texto, rotation=30, labelsize=8)
    eixos.tick_params(axis='y', colors=cor_texto, labelsize=8)
    for spine in eixos.spines.values():
        spine.set_edgecolor(cor_texto)
    
    figura_matplotlib.tight_layout()
    buf = io.BytesIO()
    figura_matplotlib.savefig(buf, format="png", transparent=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def atualizar_elementos_ui(page: ft.Page):
    indicador_carregamento.visible = True
    page.update()

    sentinela.run_analysis() # Roda a análise completa

    # Atualiza valores de gases
    if sentinela.latest_reading_data is not None:
        for nome_gas in nomes_gases_configurados:
            if nome_gas in mapa_textos_valores_gases:
                valor = sentinela.latest_reading_data.get(nome_gas, 0)
                mapa_textos_valores_gases[nome_gas].value = f"{valor:.2f} ppm"
        
        # Atualiza ambiente
        texto_umidade.value = f"Umidade: {sentinela.latest_reading_data.get('Umidade_Relativa_percent', 0):.1f}%"
        texto_temperatura.value = f"Temperatura: {sentinela.latest_reading_data.get('Temperatura_C', 0):.1f}°C"
        
        # Atualiza qualidade do ar
        qualidade = sentinela.latest_reading_dt_classification or sentinela.latest_reading_rules_classification or "Indisponível"
        texto_qualidade_ar.value = qualidade
        mapa_feedback = {'Excelente': ft.colors.GREEN, 'Bom': ft.colors.LIGHT_GREEN, 'Regular': ft.colors.AMBER, 'Ruim': ft.colors.ORANGE, 'Muito Ruim': ft.colors.RED, 'Crítico': ft.colors.PURPLE}
        container_qualidade_ar.bgcolor = mapa_feedback.get(qualidade, ft.colors.BLUE_GREY)

    # Atualiza gráfico
    controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(page.theme_mode)

    indicador_carregamento.visible = False
    page.update()

def exibir_relatorio(page: ft.Page, tipo: str):
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
    """Loop que executa em uma thread para atualizações automáticas."""
    while not parar_thread_atualizacao.is_set():
        if switch_atualizacao_automatica.value:
            try:
                # O Flet gerencia o estado da UI, então podemos chamar a atualização diretamente
                # desde que as operações no backend sejam thread-safe (leitura de arquivo é).
                atualizar_elementos_ui(page)
                print("Atualização automática executada.")
            except Exception as e:
                print(f"Erro na atualização automática: {e}")
        time.sleep(10) # Intervalo de atualização

def main(page: ft.Page):
    page.title = "Sentinela Verde Ambiental"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.fonts = {"Consolas": "Consolas, 'Courier New', monospace"}
    
    # Inicia a thread de atualização em background
    update_thread = threading.Thread(target=realtime_update_loop, args=(page,), daemon=True)
    update_thread.start()
    
    # Função para parar a thread quando a aplicação fechar
    def on_disconnect(e):
        parar_thread_atualizacao.set()
    
    page.on_disconnect = on_disconnect
    
    # --- UI ---
    
    # Cards de Gases
    cards_gases = []
    for i, nome_gas in enumerate(nomes_gases_configurados):
        texto_valor = ft.Text("-- ppm", weight=ft.FontWeight.BOLD, size=20)
        mapa_textos_valores_gases[nome_gas] = texto_valor
        cards_gases.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(nome_gas.replace('_ppm', ''), weight=ft.FontWeight.BOLD),
                        texto_valor
                    ]),
                    padding=15
                ), expand=True
            )
        )
    
    # Painel de Qualidade do Ar
    container_qualidade_ar.content = ft.Column([
        ft.Row([icone_qualidade_ar, ft.Text("Qualidade do Ar")]),
        texto_qualidade_ar
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    container_qualidade_ar.padding = 15
    container_qualidade_ar.border_radius = 10
    
    # Botões
    botao_atualizar = ft.FilledButton("Atualizar Dados", icon=ft.icons.REFRESH, on_click=lambda e: atualizar_elementos_ui(page))
    
    # --- Abas ---
    
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
    
    conteudo_aba_relatorios = ft.Column([
        ft.Row([
            ft.ElevatedButton("Resumo", on_click=lambda e: exibir_relatorio(page, "resumo")),
            ft.ElevatedButton("Árvore de Decisão", on_click=lambda e: exibir_relatorio(page, "arvore"), disabled=not PIL_DISPONIVEL),
            ft.ElevatedButton("Log", on_click=lambda e: exibir_relatorio(page, "log")),
        ]),
        ft.Container(
            content=ft.Column([area_texto_relatorio, area_imagem_relatorio], scroll=ft.ScrollMode.ADAPTIVE, expand=True),
            border=ft.border.all(1, ft.colors.OUTLINE),
            border_radius=10,
            padding=10,
            expand=True
        )
    ], expand=True)

    abas = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Monitoramento", content=conteudo_aba_monitoramento),
            ft.Tab(text="Relatórios", content=conteudo_aba_relatorios),
        ],
        expand=True,
    )

    page.add(abas)
    atualizar_elementos_ui(page) # Carga inicial

ft.app(target=main)