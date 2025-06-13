# frontend.py (versão corrigida que recebe a instância do backend)
# -*- coding: utf-8 -*-

import flet as ft
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
from typing import Dict, Any

# Importa o backend apenas para anotação de tipo, evitando importação circular.
from backend import SentinelaVerde

# --- Referências Globais a Controles Flet ---
# Manter referências globais facilita a atualização a partir de funções de callback.
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

def gerar_imagem_grafico_base64(summary_data: Dict[str, Any], page_theme_mode: ft.ThemeMode) -> str:
    """Cria um gráfico de previsão e o retorna como uma string base64."""
    figura_matplotlib = Figure(figsize=(7, 3.8), dpi=100)
    eixos = figura_matplotlib.add_subplot(111)
    cor_texto = 'white' if page_theme_mode == ft.ThemeMode.DARK else 'black'
    figura_matplotlib.set_facecolor('none')
    eixos.set_facecolor('none')

    

    df_forecast = summary_data.get('previsoes')
    tem_dados = False
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
    # A linha abaixo foi REMOVIDA para corrigir o erro 'AttributeError'.
    # if not page.session or not page.session.id: return
    
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
    
    controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(summary, page.theme_mode)

    indicador_carregamento.visible = False
    page.update()

def main(page: ft.Page, sentinela_instance: 'SentinelaVerde'):
    """
    Constrói a interface gráfica.
    Esta função RECEBE a instância do 'sentinela' do main_app.py, não a cria.
    """
    page.title = "Sentinela Verde Ambiental"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START
    
    sentinela_instance.page_update_callback = lambda: atualizar_elementos_ui(page, sentinela_instance)

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
        [ft.Row([icone_qualidade_ar, ft.Text("Qualidade do Ar (IA)")]), texto_qualidade_ar],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )
    container_qualidade_ar.padding = 15
    container_qualidade_ar.border_radius = 10
    
    layout = ft.Column([
        ft.Row([card_concentracao, card_pm25, card_pm10], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([
            ft.Card(content=ft.Container(controle_imagem_plot, padding=10), expand=3),
            ft.Column([
                ft.Card(content=ft.Container(ft.Column([texto_temperatura, texto_umidade]), padding=15)),
                ft.Card(content=container_qualidade_ar, expand=True)
            ], expand=1)
        ]),
        ft.Row([indicador_carregamento], alignment=ft.MainAxisAlignment.CENTER)
    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True, spacing=10)

    page.add(ft.Container(content=layout, padding=15, expand=True))
    
    sentinela_instance.run_analysis()
