# frontend_final.py (versão mesclada e corrigida)
# -*- coding: utf-8 -*-

import flet as ft
import pandas as pd
import matplotlib
import io
import base64
import threading
import requests
import os
import time
from typing import Optional, Dict, Any

matplotlib.use('Agg')
from matplotlib.figure import Figure

# Importa o backend apenas para anotação de tipo, evitando importação circular.
from backend import SentinelaVerde

# --- Configurações Globais e Chave de API ---
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN",
                           "845535b9a4be3918e97e326f4c550afad64b21d5")

# --- Referências Globais a Controles Flet ---
# Controles do Frontend2 (base)
texto_concentracao_geral = ft.Text("-- ppm", weight=ft.FontWeight.BOLD, size=20)
texto_pm25 = ft.Text("-- µg/m³", weight=ft.FontWeight.BOLD, size=20)
texto_pm10 = ft.Text("-- µg/m³", weight=ft.FontWeight.BOLD, size=20)
texto_umidade = ft.Text("...")
texto_temperatura = ft.Text("...")
icone_qualidade_ar = ft.Icon(ft.Icons.HELP_OUTLINE, size=24)
texto_qualidade_ar = ft.Text("Aguardando...", size=18, weight=ft.FontWeight.BOLD)
container_qualidade_ar = ft.Container()
controle_imagem_plot = ft.Image(fit=ft.ImageFit.CONTAIN, expand=True)
indicador_carregamento = ft.ProgressRing(visible=False, width=24, height=24)
icone_qualidade_previsao = ft.Icon(ft.Icons.HELP_OUTLINE, size=20)
texto_qualidade_previsao = ft.Text("Previsão (24h): ...", weight=ft.FontWeight.BOLD)

# Controles do Frontend1 (restaurados)
area_texto_relatorio = ft.Text("", expand=True, selectable=True, font_family="Consolas")
area_imagem_relatorio = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, expand=True)
switch_atualizacao_automatica = ft.Switch(label="Atualização em tempo real", value=False)
parar_thread_atualizacao = threading.Event()

dropdown_sensores_externos = ft.Dropdown(
    options=[ft.dropdown.Option(key="rio claro", text="Rio Claro")],
    hint_text="Selecione uma cidade",
    expand=True
)
texto_qualidade_sensor_externo = ft.Text("Consulta em tempo real.", size=14, weight=ft.FontWeight.BOLD)
indicador_carregamento_externo = ft.ProgressRing(visible=False, width=16, height=16)
container_sensores_externos = ft.Container()


def gerar_imagem_grafico_base64(df_forecast: Optional[pd.DataFrame], page_theme_mode: ft.ThemeMode) -> str:
    """Cria um gráfico de previsão e o retorna como uma string base64."""
    figura_matplotlib = Figure(figsize=(7, 3.8), dpi=100)
    eixos = figura_matplotlib.add_subplot(111)
    cor_texto = 'white' if page_theme_mode == ft.ThemeMode.DARK else 'black'
    figura_matplotlib.set_facecolor('none')
    eixos.set_facecolor('none')

    tem_dados = False
    # AQUI ESTÁ A CORREÇÃO: Ele agora usa a variável 'df_forecast' que foi passada como argumento
    if df_forecast is not None and not df_forecast.empty:
        for col in df_forecast.columns:
            if pd.api.types.is_numeric_dtype(df_forecast[col]):
                tem_dados = True
                label = col.replace('_PPM', ' PPM').replace('_ug_m3', ' µg/m³')
                eixos.plot(df_forecast.index, df_forecast[col], label=label)

    if tem_dados:
        eixos.legend(prop={'size': 8}, labelcolor=cor_texto)
        eixos.set_title("Previsão (Próximas 24h)", color=cor_texto, size=10)
    else:
        eixos.text(0.5, 0.5, "Dados insuficientes para previsão", ha='center', va='center', color=cor_texto)

    eixos.tick_params(axis='x', colors=cor_texto, rotation=30, labelsize=8)
    eixos.tick_params(axis='y', colors=cor_texto, labelsize=8)
    for spine in eixos.spines.values():
        spine.set_edgecolor(cor_texto)

    figura_matplotlib.tight_layout(pad=1.5)
    buf = io.BytesIO()
    figura_matplotlib.savefig(buf, format="png", transparent=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def atualizar_elementos_ui(page: ft.Page, sentinela_instance: 'SentinelaVerde'):
    """Função central que ATUALIZA a UI com os dados mais recentes do backend."""
    indicador_carregamento.visible = True
    page.update()

    summary = sentinela_instance.get_latest_data_summary()

    if summary:
        conc_geral = summary.get('Concentracao_Geral_PPM')
        texto_concentracao_geral.value = f"{conc_geral:.2f} ppm" if conc_geral is not None else "-- ppm"

        pm25 = summary.get('PM2.5_ug_m3')
        texto_pm25.value = f"{pm25:.1f} µg/m³" if pm25 is not None else "-- µg/m³"

        pm10 = summary.get('PM10_ug_m3')
        texto_pm10.value = f"{pm10:.1f} µg/m³" if pm10 is not None else "-- µg/m³"

        umid = summary.get('Umidade_Relativa_percent')
        texto_umidade.value = f"Umidade: {umid:.1f}%" if umid is not None else "Umidade: --"

        temp = summary.get('Temperatura_C')
        texto_temperatura.value = f"Temperatura: {temp:.1f}°C" if temp is not None else "Temperatura: --"

        qualidade = summary.get('qualidade_ar', "Indisponível")
        texto_qualidade_ar.value = qualidade.upper()

        mapa_feedback = {
            'Excelente': (ft.Icons.SENTIMENT_VERY_SATISFIED, ft.Colors.GREEN_400),
            'Bom': (ft.Icons.SENTIMENT_SATISFIED, ft.Colors.LIGHT_GREEN_600),
            'Ruim': (ft.Icons.SENTIMENT_DISSATISFIED, ft.Colors.ORANGE_700),
            'IA não treinada': (ft.Icons.COMPUTER, ft.Colors.BLUE_GREY_500),
            'Analisando...': (ft.Icons.HOURGLASS_EMPTY, ft.Colors.BLUE_GREY_500),
            'Aguardando...': (ft.Icons.HOURGLASS_EMPTY, ft.Colors.BLUE_GREY_500)
        }
        icone, cor = mapa_feedback.get(qualidade, (ft.Icons.HELP_OUTLINE, ft.Colors.GREY))
        icone_qualidade_ar.name = icone
        container_qualidade_ar.bgcolor = cor

        # A referência a 'sentinela' foi trocada por 'sentinela_instance' para consistência
        summary_para_previsao = sentinela_instance.get_latest_data_summary()
        qualidade_prevista = summary_para_previsao.get('qualidade_previsao', 'Indisponível')
        texto_qualidade_previsao.value = f"Previsão (24h): {qualidade_prevista}"

        mapa_feedback_previsao = {
            'Bom': (ft.Icons.THUMB_UP_OFF_ALT, ft.Colors.GREEN),
            'Ruim': (ft.Icons.THUMB_DOWN_OFF_ALT, ft.Colors.ORANGE_800),
            'Insuficiente': (ft.Icons.HOURGLASS_EMPTY, ft.Colors.BLUE_GREY),
        }

        icone, cor = mapa_feedback_previsao.get(qualidade_prevista, (ft.Icons.HELP_OUTLINE, ft.Colors.GREY))
        icone_qualidade_previsao.name = icone
        icone_qualidade_previsao.color = cor
        texto_qualidade_previsao.color = cor

    # ***** LINHA CORRIGIDA AQUI *****
    # Agora passamos o DataFrame da previsão (acessado via sentinela_instance.df_forecast)
    # para a função que gera o gráfico.
    controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(sentinela_instance.future_forecast, page.theme_mode)

    indicador_carregamento.visible = False
    page.update()


# --- Funções Restauradas do Frontend1 ---

def fetch_and_update_waqi_data(e):
    cidade = e.control.value
    if not cidade: return

    texto_qualidade_sensor_externo.value = f"Consultando {cidade.title()}..."
    texto_qualidade_sensor_externo.color = None
    indicador_carregamento_externo.visible = True
    if container_sensores_externos.page: container_sensores_externos.update()

    try:
        url = f"https://api.waqi.info/feed/{cidade}/?token={WAQI_API_TOKEN}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ok":
            aqi = data["data"]["aqi"]
            if aqi <= 50:
                qualidade, cor = "Boa", ft.Colors.LIGHT_GREEN_700
            elif aqi <= 100:
                qualidade, cor = "Moderada", ft.Colors.AMBER_700
            elif aqi <= 150:
                qualidade, cor = "Ruim (Grupos Sensíveis)", ft.Colors.ORANGE_800
            elif aqi <= 200:
                qualidade, cor = "Ruim", ft.Colors.RED_700
            else:
                qualidade, cor = "Perigosa", ft.Colors.PURPLE_800
            texto_qualidade_sensor_externo.value = f"AQI: {aqi} ({qualidade})"
        else:
            texto_qualidade_sensor_externo.value = f"Erro: {data.get('data', 'Não encontrado')}"
            cor = ft.Colors.ERROR
        texto_qualidade_sensor_externo.color = cor
    except requests.exceptions.RequestException:
        texto_qualidade_sensor_externo.value = "Erro de conexão com a API."
        texto_qualidade_sensor_externo.color = ft.Colors.ERROR
    finally:
        indicador_carregamento_externo.visible = False
        if container_sensores_externos.page: container_sensores_externos.update()


def consultar_sensor_externo_thread(e):
    threading.Thread(target=fetch_and_update_waqi_data, args=(e,), daemon=True).start()


def exibir_resumo(page: ft.Page, sentinela_instance: 'SentinelaVerde'):
    area_texto_relatorio.value = sentinela_instance.get_formatted_summary()  # Usa a instância
    area_texto_relatorio.visible = True
    area_imagem_relatorio.visible = False
    page.update()


def exibir_recomendacoes(page: ft.Page):
    qualidade_atual = texto_qualidade_ar.value
    mapa_recomendacoes = {
        "EXCELENTE": "Aproveite o ar livre! As condições são ideais para atividades externas.",
        "BOM": "Qualidade do ar aceitável. Pessoas muito sensíveis devem considerar limitar esforços prolongados ao ar livre.",
        "RUIM": "Grupos sensíveis (crianças, idosos, pessoas com doenças respiratórias) devem evitar atividades ao ar livre. A população em geral deve reduzir esforço prolongado.",
        "IA NÃO TREINADA": "O modelo de Inteligência Artificial ainda não possui dados suficientes para classificar a qualidade do ar.",
        "INDISPONÍVEL": "Não foi possível determinar a qualidade do ar no momento."
    }
    texto_recomendacao = mapa_recomendacoes.get(qualidade_atual, "Recomendações não disponíveis.")
    area_texto_relatorio.value = f"--- Recomendações para Qualidade do Ar: {qualidade_atual} ---\n\n{texto_recomendacao}"
    area_texto_relatorio.visible = True
    area_imagem_relatorio.visible = False
    page.update()


def realtime_update_loop(page: ft.Page, sentinela_instance: 'SentinelaVerde'):
    while not parar_thread_atualizacao.is_set():
        if switch_atualizacao_automatica.value:
            try:
                # Chama a função de atualização passando a instância
                atualizar_elementos_ui(page, sentinela_instance)
            except Exception as e:
                print(f"Erro na atualização automática: {e}")

        time.sleep(5)


def main(page: ft.Page, sentinela_instance: 'SentinelaVerde'):
    """Constrói a interface gráfica completa, mesclando as duas versões."""
    page.title = "Sentinela Verde Ambiental"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.fonts = {"Consolas": "Consolas, 'Courier New', monospace"}

    # --- Configura o callback e inicia a thread de atualização ---
    sentinela_instance.page_update_callback = lambda: atualizar_elementos_ui(page, sentinela_instance)

    update_thread = threading.Thread(target=realtime_update_loop, args=(page, sentinela_instance), daemon=True)
    update_thread.start()
    page.on_disconnect = lambda e: parar_thread_atualizacao.set()

    # --- Definição dos Componentes da UI ---
    card_concentracao = ft.Card(content=ft.Container(
        content=ft.Column([ft.Text("Concentração Gases (p.p.m)"), texto_concentracao_geral]), padding=15
    ))
    card_pm25 = ft.Card(content=ft.Container(
        content=ft.Column([ft.Text("PM2.5"), texto_pm25]), padding=15
    ))
    card_pm10 = ft.Card(content=ft.Container(
        content=ft.Column([ft.Text("PM10"), texto_pm10]), padding=15
    ))

    container_qualidade_ar.content = ft.Column(
        [ft.Row([icone_qualidade_ar, ft.Text("Qualidade do Ar (IA Local)")]), texto_qualidade_ar],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    container_qualidade_ar.padding = 15
    container_qualidade_ar.border_radius = 10

    # Container de sensores externos (ligado à função real)
    dropdown_sensores_externos.on_change = consultar_sensor_externo_thread
    container_sensores_externos.content = ft.Column([
        ft.Row([ft.Icon(ft.Icons.TRAVEL_EXPLORE), ft.Text("Consulta Externa (WAQI)")]),
        dropdown_sensores_externos,
        ft.Row([indicador_carregamento_externo, texto_qualidade_sensor_externo],
               vertical_alignment=ft.CrossAxisAlignment.CENTER)
    ])
    container_sensores_externos.padding = 15

    botao_atualizar = ft.FilledButton("Atualizar Dados", icon=ft.Icons.REFRESH,
                                      on_click=lambda e: atualizar_elementos_ui(page, sentinela_instance))

    # --- Aba de Monitoramento ---
    conteudo_aba_monitoramento = ft.Column([
        ft.Row([card_concentracao, card_pm25, card_pm10], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Previsão Futura dos Níveis de Gás", weight=ft.FontWeight.BOLD, size=16),
                        controle_imagem_plot,
                        ft.Container(height=5),
                        ft.Row(
                            [icone_qualidade_previsao, texto_qualidade_previsao],
                            alignment=ft.MainAxisAlignment.CENTER,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=8
                        )
                    ]),
                    padding=15
                ), expand=3
            ),
            ft.Column([
                ft.Card(content=ft.Container(ft.Column([texto_temperatura, texto_umidade]), padding=15)),
                ft.Card(content=container_qualidade_ar, expand=True),
                ft.Card(content=container_sensores_externos, expand=True)
            ], expand=1, spacing=10)
        ]),
        ft.Row([botao_atualizar, switch_atualizacao_automatica, indicador_carregamento],
               alignment=ft.MainAxisAlignment.CENTER)
    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True, spacing=10, alignment=ft.MainAxisAlignment.START)

    # --- Aba de Relatórios ---
    conteudo_aba_relatorios = ft.Column([
        ft.Row([
            ft.ElevatedButton("Resumo", on_click=lambda e: exibir_resumo(page, sentinela_instance)),
            ft.ElevatedButton("Recomendações", icon=ft.Icons.RECOMMEND, on_click=lambda e: exibir_recomendacoes(page)),
        ]),
        ft.Container(
            content=ft.Column([area_texto_relatorio, area_imagem_relatorio], scroll=ft.ScrollMode.ADAPTIVE,
                              expand=True),
            border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=10, padding=10, expand=True
        )
    ], expand=True)

    # --- Estrutura de Abas ---
    abas = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Monitoramento", content=ft.Container(conteudo_aba_monitoramento, padding=15)),
            ft.Tab(text="Relatórios", content=ft.Container(conteudo_aba_relatorios, padding=15)),
        ],
        expand=True,
    )

    page.add(abas)

    # Inicia a primeira análise e atualização da UI
    sentinela_instance.run_analysis()
