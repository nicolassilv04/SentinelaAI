# frontend_flet_final_comentado.py

# --- Importações Necessárias ---
import flet as ft  # Biblioteca principal do Flet para a interface gráfica
import matplotlib  # Para criar gráficos estáticos

matplotlib.use('Agg')  # Configura Matplotlib para não usar um backend de GUI interativo
import matplotlib.pyplot as plt  # Módulo de plotagem do Matplotlib
from matplotlib.figure import Figure  # Classe para criar a figura do gráfico
import numpy as np  # Para manipulação numérica, especialmente para o gráfico
import io  # Para manipular dados binários em memória (streams de bytes)
import base64  # Para codificar a imagem do gráfico em base64 para o Flet
import backend  # Seu módulo backend com a lógica do sistema SentinelaVerde

# Tenta importar a biblioteca Pillow (PIL) para manipulação de imagens (árvore de decisão)
try:
    from PIL import Image as PILImage  # Renomeia Image para PILImage para evitar conflito com ft.Image

    PIL_DISPONIVEL = True  # Flag indicando que Pillow está disponível
except ImportError:
    PIL_DISPONIVEL = False  # Flag indicando que Pillow não está disponível
    PILImage = None  # Define PILImage como None para evitar NameError se não importado

# --- Configurações Iniciais e Instância do Backend ---
# Cria dados de exemplo (se a função existir e for necessária no seu backend)
backend.create_sample_data()
# Cria a instância principal do seu sistema de backend
sentinela = backend.SentinelaVerde()

# --- Configurações da Figura Matplotlib e Cache de Dados Globais ---
# Cria a figura e os eixos do Matplotlib uma vez; serão reutilizados e atualizados
figura_matplotlib = Figure(figsize=(7, 3.8), dpi=90, facecolor='white')  # Dimensões e cor de fundo da figura
eixos_matplotlib = figura_matplotlib.add_subplot(111)  # Adiciona uma área de plotagem à figura
string_base64_imagem_plot = ""  # Armazenará a string base64 da imagem do gráfico renderizada

# Dicionários para armazenar em cache os dados buscados do backend, otimizando atualizações
cache_dados_gases = {}
cache_dados_ambientais = {}

# Obtém nomes dos gases e define paletas de cores para a UI
nomes_gases_configurados = sentinela.config.get('gas_forecasting', {}).get('target_gas_columns', [])
cores_gases_painel_hex = ['#e57373', '#64b5f6', '#81c784', '#ffd54f']  # Cores para os cards de gás
cores_gases_plot_hex = ['#d32f2f', '#1976d2', '#388e3c', '#fbc02d']  # Cores para as linhas no gráfico

# Dicionário para guardar o estado dos checkboxes de filtro do gráfico
estado_filtro_gases_plot = {}

# Limites de exemplo para alertas visuais nos cards de gás
limites_alerta_gases = {
    "CO2_ppm": 1000, "CH4_ppm": 50, "NH3_ppm": 25, "VOC_ppm": 10
}

# --- Referências Globais aos Controles Flet ---
# Manter referências globais a controles que precisam ser atualizados dinamicamente
# simplifica o acesso em diferentes funções no escopo deste app Flet.

# Dicionário para referências aos ft.Text que exibem os valores dos gases
mapa_textos_valores_gases = {}
# Dicionário para referências aos ft.Icon de alerta dos gases
mapa_alertas_icones_gases = {}

# Controles para dados ambientais
texto_umidade = ft.Text("Umidade: --%", size=14, color=ft.Colors.WHITE)
texto_temperatura = ft.Text("Temperatura: --°C", size=14, color=ft.Colors.WHITE)

# Controles para qualidade do ar (ícone, texto e o container que muda de cor)
icone_qualidade_ar = ft.Icon(ft.Icons.HELP_OUTLINE, color=ft.Colors.WHITE, size=24)
texto_qualidade_ar = ft.Text("--", weight=ft.FontWeight.BOLD, size=18, color=ft.Colors.WHITE)
container_qualidade_ar = ft.Container(
    content=ft.Column([
        ft.Row([icone_qualidade_ar,
                ft.Text(" Qualidade do Ar", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.WHITE)],
               alignment=ft.MainAxisAlignment.CENTER),
        texto_qualidade_ar
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
    alignment=ft.alignment.center, expand=True, padding=15, border_radius=10
)

# Controle para exibir a imagem do gráfico Matplotlib
controle_imagem_plot = ft.Image(src_base64=string_base64_imagem_plot, fit=ft.ImageFit.CONTAIN, expand=True)

# Controles para a área de exibição de relatórios
area_texto_relatorio = ft.Text("", expand=True, selectable=True, size=13,
                               font_family="Consolas, 'Courier New', monospace")
area_imagem_relatorio = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN, expand=True)
# Coluna que agrupa texto e imagem do relatório, com scroll
coluna_conteudo_relatorio = ft.Column(
    [area_texto_relatorio, area_imagem_relatorio],
    expand=True, scroll=ft.ScrollMode.ADAPTIVE, spacing=10
)

# Indicador de carregamento (círculo de progresso)
indicador_carregamento = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3,
                                         color=ft.Colors.BLUE_ACCENT,  # Cor primária do anel
                                         bgcolor=ft.Colors.with_opacity(0.2,
                                                                        ft.Colors.BLUE_ACCENT))  # Cor de fundo suave
# Lista para os checkboxes de filtro do gráfico (será populada em main)
checkboxes_filtro_gases = []


# --- Função Auxiliar para Gerar Imagem do Gráfico Matplotlib ---
def gerar_imagem_grafico_base64(page_theme_mode=ft.ThemeMode.LIGHT):
    """
    Renderiza o gráfico Matplotlib com os dados e tema atuais,
    retornando a imagem como uma string base64.
    """
    global string_base64_imagem_plot
    eixos_matplotlib.clear()
    horas_plot = list(range(0, 25, 4))

    # Define cores do gráfico baseadas no tema da página (claro/escuro)
    if page_theme_mode == ft.ThemeMode.DARK:
        cor_fundo_eixo_mpl = (0.18, 0.18, 0.18, 1.0)  # RGBA para Matplotlib
        cor_fundo_figura_mpl = (0.12, 0.12, 0.12, 1.0)
        cor_texto_grafico_mpl = 'white'
        cor_grid_mpl_matplotlib = (0.5, 0.5, 0.5, 0.5)
    else:  # Tema Claro
        cor_fundo_eixo_mpl = '#f9f9f9'  # Hexadecimal para Matplotlib
        cor_fundo_figura_mpl = 'white'
        cor_texto_grafico_mpl = '#333333'
        cor_grid_mpl_matplotlib = 'lightgrey'

    figura_matplotlib.set_facecolor(cor_fundo_figura_mpl)  # Cor de fundo da figura inteira
    eixos_matplotlib.set_facecolor(cor_fundo_eixo_mpl)  # Cor de fundo da área de plotagem

    tem_dados_para_plotar = False
    for i, nome_gas_col in enumerate(nomes_gases_configurados):
        if i >= 4: break
        if estado_filtro_gases_plot.get(nome_gas_col, True):  # Verifica se o gás está selecionado no filtro
            previsao_gas = cache_dados_gases.get(nome_gas_col, {}).get('previsao', [])
            if previsao_gas:
                tem_dados_para_plotar = True
                eixos_matplotlib.plot(horas_plot, previsao_gas, marker='o', linestyle='-',
                                      label=nome_gas_col.replace('_ppm', ''),
                                      color=cores_gases_plot_hex[i % len(cores_gases_plot_hex)],
                                      linewidth=2.5, markersize=5)

    if tem_dados_para_plotar:
        leg = eixos_matplotlib.legend(fontsize=9)
        for text in leg.get_texts():  # Ajusta cor do texto da legenda para o tema
            text.set_color(cor_texto_grafico_mpl)
    else:
        eixos_matplotlib.text(0.5, 0.5, "Selecione gases para exibir\nou sem dados de previsão.",
                              ha='center', va='center', fontsize=10,
                              color=cor_texto_grafico_mpl if page_theme_mode == ft.ThemeMode.DARK else 'grey')

    # Configurações visuais do gráfico
    eixos_matplotlib.grid(True, which='both', linestyle=':', linewidth=0.7, color=cor_grid_mpl_matplotlib)
    eixos_matplotlib.set_title("Previsão de Concentração de Gases (Próximas 24h)", fontsize=12, weight='bold',
                               color=cor_texto_grafico_mpl)
    eixos_matplotlib.set_xlabel("Horas a partir de agora", fontsize=10, color=cor_texto_grafico_mpl)
    eixos_matplotlib.set_ylabel("Concentração (ppm)", fontsize=10, color=cor_texto_grafico_mpl)
    eixos_matplotlib.tick_params(axis='x', colors=cor_texto_grafico_mpl, labelsize=9)
    eixos_matplotlib.tick_params(axis='y', colors=cor_texto_grafico_mpl, labelsize=9)

    # Ajusta a cor das bordas (spines) do gráfico para o tema
    for spine in eixos_matplotlib.spines.values():
        spine.set_edgecolor(cor_texto_grafico_mpl)

    figura_matplotlib.tight_layout(pad=1.0)  # Evita que elementos do gráfico sejam cortados

    # Salva o gráfico em um buffer de bytes em memória
    buffer_imagem = io.BytesIO()
    figura_matplotlib.savefig(buffer_imagem, format="png", bbox_inches='tight',
                              facecolor=figura_matplotlib.get_facecolor())
    buffer_imagem.seek(0)  # Volta para o início do buffer
    # Codifica a imagem para base64
    string_base64_imagem_plot = base64.b64encode(buffer_imagem.getvalue()).decode("utf-8")
    return string_base64_imagem_plot


# --- Funções de Lógica e Atualização da Interface do Usuário (UI) ---
def buscar_e_cachear_dados():
    """Busca os dados mais recentes do backend e os armazena nos caches globais."""
    global cache_dados_gases, cache_dados_ambientais, nomes_gases_configurados
    sentinela.run_analysis()  # Ponto principal de chamada da lógica do backend
    cache_dados_gases, cache_dados_ambientais = {}, {}  # Limpa os caches antes de popular

    if sentinela.latest_reading_data is not None:  # Se houver dados de leitura
        for nome_gas_col in nomes_gases_configurados:
            cache_dados_gases[nome_gas_col] = {'atual': sentinela.latest_reading_data.get(nome_gas_col, 0)}
        cache_dados_ambientais.update({
            'umidade': sentinela.latest_reading_data.get('Umidade_Relativa_percent', 0),
            'temperatura': sentinela.latest_reading_data.get('Temperatura_C', 0),
            'qualidade_ar': sentinela.latest_reading_dt_classification or sentinela.latest_reading_rules_classification or "Indisponível"
        })
    else:  # Valores padrão se não houver dados da última leitura
        cache_dados_ambientais.update({'umidade': 0, 'temperatura': 0, 'qualidade_ar': "Indisponível"})
        for nome_gas_col in nomes_gases_configurados: cache_dados_gases[nome_gas_col] = {'atual': 0}

    # Preenche o cache com dados de previsão
    for nome_gas_col in nomes_gases_configurados:
        if nome_gas_col not in cache_dados_gases: cache_dados_gases[
            nome_gas_col] = {}  # Garante que a chave do gás exista
        # Define um valor padrão para a previsão (repetir o valor atual 7 vezes)
        cache_dados_gases[nome_gas_col]['previsao'] = [cache_dados_gases[nome_gas_col].get('atual', 0)] * 7
        # Se houver dados de previsão do backend, usa-os
        if sentinela.df_future_gases_forecast is not None and nome_gas_col in sentinela.df_future_gases_forecast.columns:
            previsao_completa = sentinela.df_future_gases_forecast[nome_gas_col].tolist()
            if previsao_completa:  # Se a lista de previsão não estiver vazia
                indices = np.linspace(0, len(previsao_completa) - 1, 7, dtype=int)  # Pega 7 pontos espaçados
                cache_dados_gases[nome_gas_col]['previsao'] = [previsao_completa[i] for i in indices] if len(
                    indices) > 0 else [cache_dados_gases[nome_gas_col].get('atual', 0)] * 7


def atualizar_elementos_ui(page: ft.Page):
    """Atualiza todos os elementos visuais da página com os dados dos caches."""
    indicador_carregamento.visible = True  # Mostra o anel de progresso
    if page: page.update(indicador_carregamento)  # Atualiza apenas o indicador para feedback rápido

    buscar_e_cachear_dados()  # Busca e processa os dados mais recentes

    # Atualiza os textos dos valores e alertas dos gases
    for nome_gas_col in nomes_gases_configurados:
        if nome_gas_col in mapa_textos_valores_gases:  # Verifica se o controle de UI existe
            valor_atual = cache_dados_gases.get(nome_gas_col, {}).get('atual', 0)
            mapa_textos_valores_gases[nome_gas_col].value = f"{valor_atual:.2f} ppm"

            limite = limites_alerta_gases.get(nome_gas_col)  # Pega o limite de alerta (hardcoded)
            # Lógica para exibir alerta visual se o valor ultrapassar o limite
            if limite is not None and valor_atual > limite:
                mapa_alertas_icones_gases[nome_gas_col].visible = True
                mapa_textos_valores_gases[nome_gas_col].color = ft.Colors.RED_ACCENT_700  # Destaca valor em vermelho
            else:
                mapa_alertas_icones_gases[nome_gas_col].visible = False
                # Define a cor do texto do valor do gás de acordo com o tema da página
                mapa_textos_valores_gases[
                    nome_gas_col].color = ft.Colors.ON_SURFACE if page.theme_mode == ft.ThemeMode.DARK else ft.Colors.BLACK87

    # Atualiza textos de umidade e temperatura
    texto_umidade.value = f"Umidade: {cache_dados_ambientais.get('umidade', 0):.1f}%"
    texto_temperatura.value = f"Temperatura: {cache_dados_ambientais.get('temperatura', 0):.1f}°C"

    # Atualiza o painel de qualidade do ar (texto, ícone e cor de fundo)
    qualidade = cache_dados_ambientais.get('qualidade_ar', 'Indisponível')
    texto_qualidade_ar.value = qualidade

    mapa_feedback_qualidade = {  # Mapeamento de status para ícones e cores
        'Excelente': (ft.Icons.SENTIMENT_VERY_SATISFIED_ROUNDED, ft.Colors.GREEN_ACCENT_700),
        'Bom': (ft.Icons.SENTIMENT_SATISFIED_ROUNDED, ft.Colors.LIGHT_GREEN_700),
        'Regular': (ft.Icons.SENTIMENT_NEUTRAL_ROUNDED, ft.Colors.AMBER_700),
        'Ruim': (ft.Icons.SENTIMENT_DISSATISFIED_ROUNDED, ft.Colors.ORANGE_900),
        'Muito Ruim': (ft.Icons.SENTIMENT_VERY_DISSATISFIED_ROUNDED, ft.Colors.RED_700),
        'Crítico': (ft.Icons.MOOD_BAD_ROUNDED, ft.Colors.RED_900)
    }
    icone_q, cor_q = mapa_feedback_qualidade.get(qualidade, (ft.Icons.HELP_OUTLINE_ROUNDED, ft.Colors.BLUE_GREY_700))
    icone_qualidade_ar.name = icone_q  # Atualiza o ícone exibido
    container_qualidade_ar.bgcolor = cor_q  # Atualiza a cor de fundo do container

    # Lógica para exibir um Banner de Alerta se a qualidade do ar for ruim ou crítica
    if qualidade in ["Ruim", "Muito Ruim", "Crítico"]:
        if page.banner is not None:  # Se o banner já existe, atualiza-o
            page.banner.leading.name = icone_q
            page.banner.bgcolor = cor_q
            page.banner.content.value = f"Atenção: Qualidade do ar classificada como '{qualidade}'. Recomenda-se cautela."
            page.banner.open = True
        else:  # Se não existe, cria um novo banner
            page.banner = ft.Banner(
                bgcolor=cor_q,
                leading=ft.Icon(icone_q, color=ft.Colors.WHITE, size=30),
                content=ft.Text(f"Atenção: Qualidade do ar classificada como '{qualidade}'. Recomenda-se cautela.",
                                color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
                actions=[ft.TextButton("Fechar", on_click=lambda e: fechar_banner(page))],  # Botão para fechar o banner
                open=True  # Define para abrir ao criar
            )
    elif page.banner is not None and page.banner.open:  # Se a qualidade melhorar e o banner estiver aberto, fecha-o
        page.banner.open = False

    # Regenera e atualiza a imagem do gráfico
    controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(page.theme_mode)

    indicador_carregamento.visible = False  # Esconde o anel de progresso
    if page: page.update()  # Atualiza a página inteira para refletir todas as mudanças


def fechar_banner(page: ft.Page):
    """Fecha o banner de alerta da página."""
    if page.banner:  # Verifica se o banner existe
        page.banner.open = False
        page.update()


def exibir_relatorio(page: ft.Page, tipo_relatorio: str):
    """Controla a exibição dos diferentes tipos de relatório na aba 'Relatórios'."""
    area_texto_relatorio.visible = False  # Esconde área de texto por padrão
    area_imagem_relatorio.visible = False  # Esconde área de imagem por padrão
    conteudo_texto_relatorio = ""
    caminho_imagem_arvore = "Não especificado"  # Inicializa para o caso de erro

    if tipo_relatorio == "resumo":
        conteudo_texto_relatorio = sentinela.get_formatted_summary()  # Pega resumo do backend
        area_texto_relatorio.value = conteudo_texto_relatorio
        area_texto_relatorio.visible = True
    elif tipo_relatorio == "log":
        try:
            with open('sentinela_verde.log', 'r', encoding='utf-8') as f:
                linhas_log = f.readlines()
            if not linhas_log:
                conteudo_texto_relatorio = "Log do sistema está vazio."
            else:  # Filtra ou resume o log para exibição
                palavras_chave = ["ERROR", "WARNING", "FAIL", "CRITICAL", "SUCESSO", "CONCLUÍDA", "ANÁLISE", "MODELO"]
                linhas_filtradas = [linha.strip() for linha in linhas_log if
                                    any(palavra.upper() in linha.upper() for palavra in palavras_chave) or len(
                                        linhas_log) < 50]
                conteudo_texto_relatorio = "\n".join(linhas_filtradas if linhas_filtradas else linhas_log[:50] + (
                    ["... (mais linhas no arquivo de log) ..."] if len(linhas_log) > 50 else []))
        except Exception as e:
            conteudo_texto_relatorio = f"Erro ao ler o arquivo de log: {e}"
        area_texto_relatorio.value = conteudo_texto_relatorio
        area_texto_relatorio.visible = True
    elif tipo_relatorio == "arvore":
        if not PIL_DISPONIVEL:  # Verifica se a biblioteca Pillow está instalada
            area_texto_relatorio.value = "A biblioteca Pillow (PIL) não está instalada.\nNão é possível exibir a imagem da árvore de decisão."
            area_texto_relatorio.visible = True
            if page:  # Exibe mensagem de erro rápida (SnackBar)
                snackbar_pil = ft.SnackBar(ft.Text("Erro: Pillow não instalado para ver a árvore!"), open=True,
                                           duration=3000)
                page.overlay.append(snackbar_pil);
                page.update()
            return
        try:
            caminho_imagem_arvore = sentinela.config['decision_tree']['plot_path']  # Pega caminho da imagem do backend
            img_pil = PILImage.open(caminho_imagem_arvore)  # Abre a imagem com Pillow
            img_pil.thumbnail((1020, 1440))  # Redimensiona para caber na tela
            buffer = io.BytesIO()  # Buffer em memória para a imagem
            img_pil.save(buffer, format="PNG")
            img_str_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Codifica para base64
            area_imagem_relatorio.src_base64 = img_str_base64  # Define a imagem no controle Flet
            area_imagem_relatorio.visible = True
        except FileNotFoundError:
            area_texto_relatorio.value = f"Arquivo da imagem da árvore não encontrado em: {caminho_imagem_arvore}"
            area_texto_relatorio.visible = True
        except Exception as e:  # Tratamento para outros erros ao carregar a imagem
            area_texto_relatorio.value = f"Erro ao carregar a imagem da árvore de decisão: {e}"
            area_texto_relatorio.visible = True
            if page:
                snackbar_arvore = ft.SnackBar(ft.Text(f"Erro ao carregar imagem da árvore: {e}"), open=True,
                                              duration=4000)
                page.overlay.append(snackbar_arvore);
                page.update()

    if page: page.update()  # Atualiza a página para mostrar o relatório correto


# --- Função Principal da Aplicação Flet (Ponto de Entrada) ---
def main(page: ft.Page):
    # Configurações globais da página (janela da aplicação)
    page.title = "Sentinela Verde Ambiental - Monitoramento Avançado"
    page.theme_mode = ft.ThemeMode.LIGHT  # Tema inicial é claro
    page.vertical_alignment = ft.MainAxisAlignment.START  # Alinha conteúdo ao topo
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH  # Faz conteúdo esticar horizontalmente
    page.padding = ft.padding.all(10)  # Padding geral da página
    page.banner = None  # Inicializa o atributo banner para evitar AttributeError

    # Registra fontes customizadas que podem ser usadas nos controles de Texto
    page.fonts = {
        "Consolas": "Consolas, 'Courier New', monospace"  # Para área de texto de log/relatório
    }

    # --- Função de Callback para Alternar o Tema (Claro/Escuro) ---
    def alternar_tema(e):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        botao_tema.selected = (page.theme_mode == ft.ThemeMode.DARK)  # Atualiza estado do botão

        # Ajusta a cor de fundo da AppBar conforme o tema
        if page.theme_mode == ft.ThemeMode.DARK:
            page.appbar.bgcolor = ft.Colors.with_opacity(0.1, ft.Colors.WHITE)  # Cinza escuro sutil
        else:
            page.appbar.bgcolor = ft.Colors.with_opacity(0.04, ft.Colors.BLACK)  # Quase branco, sutil

        atualizar_elementos_ui(page)  # Atualiza toda UI, incluindo gráfico, para refletir o novo tema

    # --- Criação do Botão de Alternância de Tema ---
    botao_tema = ft.IconButton(
        icon=ft.Icons.DARK_MODE_ROUNDED,  # Ícone exibido quando o tema claro está ativo (sugere mudar para escuro)
        selected=False,  # `False` indica que o tema claro está ativo (o ícone não está "selecionado")
        selected_icon=ft.Icons.LIGHT_MODE_ROUNDED,  # Ícone exibido quando o tema escuro está ativo
        icon_size=22,
        tooltip="Alternar tema claro/escuro",
        on_click=alternar_tema,  # Função chamada ao clicar
    )

    # --- Criação da AppBar (Barra no Topo da Página) ---
    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.Icons.ECO_ROUNDED, color=ft.Colors.GREEN_ACCENT_700, size=30),  # Ícone à esquerda
        leading_width=55,  # Largura da área do ícone leading
        title=ft.Text("Sentinela Verde", weight=ft.FontWeight.BOLD, size=22),  # Título da AppBar
        center_title=False,  # Alinha título à esquerda
        bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLACK),  # Cor de fundo inicial (para tema claro)
        actions=[  # Controles à direita da AppBar
            indicador_carregamento,  # Anel de progresso (geralmente invisível)
            botao_tema,  # Botão de alternar tema
            ft.VerticalDivider(width=10, color=ft.Colors.TRANSPARENT)  # Espaçador invisível
        ],
    )

    # --- Definição dos Handlers (Callbacks) para os Botões de Ação ---
    def ao_clicar_atualizar_dados(e):
        """Chamado ao clicar no botão 'Atualizar Dados'."""
        atualizar_elementos_ui(page)  # Atualiza todos os dados e a UI
        # Exibe mensagem rápida de feedback (SnackBar)
        snackbar_msg = ft.SnackBar(ft.Text("Dados e previsões atualizados!"), open=True, duration=2000)
        page.overlay.append(snackbar_msg);
        page.update()

    def ao_clicar_simular_dados(e):
        """Chamado ao clicar no botão 'Simular Novos Dados'."""
        backend.create_sample_data()  # Chama a função do backend
        atualizar_elementos_ui(page)  # Atualiza todos os dados e a UI
        snackbar_msg = ft.SnackBar(ft.Text("Novos dados simulados!"), open=True, duration=2000)
        page.overlay.append(snackbar_msg);
        page.update()

    def ao_alterar_filtro_gas(e):
        """Chamado quando o estado de um Checkbox de filtro de gás muda."""
        estado_filtro_gases_plot[
            e.control.data] = e.control.value  # Atualiza o estado do filtro (e.control.data é o nome_gas)
        controle_imagem_plot.src_base64 = gerar_imagem_grafico_base64(
            page.theme_mode)  # Regenera o gráfico com o filtro e tema
        page.update(controle_imagem_plot)  # Atualiza apenas a imagem do gráfico para performance

    # --- Construção Dinâmica dos Cards de Exibição de Gases ---
    cards_gases_linhas_layout = []
    linha_atual_de_cards = []
    mapa_icones_gases = {  # Mapeamento de nomes de gases para ícones Material Design
        "CO2_ppm": ft.Icons.CLOUD_QUEUE_ROUNDED, "CH4_ppm": ft.Icons.GAS_METER_ROUNDED,
        "NH3_ppm": ft.Icons.FILTER_HDR_ROUNDED, "VOC_ppm": ft.Icons.GRAIN_ROUNDED,
    }

    for i, nome_gas_col in enumerate(nomes_gases_configurados):
        if i >= 4: break  # Limita a 4 painéis de gás

        # Cria o ft.Text para o valor do gás, cor inicial para tema claro
        texto_valor_gas = ft.Text("-- ppm", weight=ft.FontWeight.BOLD, size=20, color=ft.Colors.BLACK87)
        mapa_textos_valores_gases[nome_gas_col] = texto_valor_gas  # Guarda referência

        # Cria o ft.Icon para alerta de valor alto (inicialmente invisível)
        icone_alerta_gas = ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_900, size=18, visible=False,
                                   tooltip="Valor acima do limite!")
        mapa_alertas_icones_gases[nome_gas_col] = icone_alerta_gas  # Guarda referência

        icone_do_gas = mapa_icones_gases.get(nome_gas_col, ft.Icons.BUBBLE_CHART_ROUNDED)  # Ícone padrão

        # Cria o Card individual para cada gás
        card_individual_gas = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([  # Linha para ícone do gás e nome do gás
                        ft.Icon(icone_do_gas, color=cores_gases_painel_hex[i % len(cores_gases_painel_hex)], size=26),
                        ft.Text(nome_gas_col.replace('_ppm', ''), weight=ft.FontWeight.BOLD, size=17,
                                color=ft.Colors.ON_SURFACE),  # Cor adaptável ao tema
                        icone_alerta_gas  # Ícone de alerta à direita
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),  # Espaça ícone e alerta
                    texto_valor_gas,  # Exibe o valor
                    ft.Text(f"Normal: < {limites_alerta_gases.get(nome_gas_col, 'N/A')} ppm", size=11,
                            color=ft.Colors.OUTLINE, italic=True, weight=ft.FontWeight.NORMAL),  # Faixa de exemplo
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                padding=ft.padding.all(15),
                border_radius=ft.border_radius.all(10),
                height=150,
                alignment=ft.alignment.center,
            ),
            elevation=4, expand=True,
        )
        linha_atual_de_cards.append(card_individual_gas)

        # Agrupa os cards em linhas de até 2
        if len(linha_atual_de_cards) == 2 or i == min(len(nomes_gases_configurados), 4) - 1:
            cards_gases_linhas_layout.append(
                ft.Row(linha_atual_de_cards, expand=True, alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                       spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)
            )
            linha_atual_de_cards = []

    # --- Criação dos Checkboxes para Filtro do Gráfico ---
    global checkboxes_filtro_gases  # Permite acesso na função de callback
    checkboxes_filtro_gases = [
        ft.Checkbox(label=nome_gas.replace('_ppm', ''), value=True, data=nome_gas, on_change=ao_alterar_filtro_gas)
        for nome_gas in nomes_gases_configurados[:4]  # Apenas para os gases que podem aparecer no gráfico
    ]
    for cb in checkboxes_filtro_gases:  # Inicializa o estado do filtro com base no valor dos checkboxes
        estado_filtro_gases_plot[cb.data] = cb.value

    # Container para os checkboxes de filtro do gráfico
    container_filtros_grafico = ft.Container(
        ft.Row(checkboxes_filtro_gases, wrap=True, spacing=5, run_spacing=0, alignment=ft.MainAxisAlignment.CENTER),
        padding=ft.padding.only(top=5, bottom=5)
    )

    # --- Construção do Conteúdo da Aba de Monitoramento ---
    conteudo_aba_monitoramento = ft.Column([
        ft.Text("Painel de Monitoramento Ambiental", weight=ft.FontWeight.BOLD, size=26, color=ft.Colors.PRIMARY),
        ft.Row([  # Linha principal da aba
            ft.Column([  # Coluna Esquerda: Painéis de Gás e Gráfico
                ft.Container(ft.Column(cards_gases_linhas_layout, spacing=12, expand=True),
                             padding=ft.padding.only(bottom=12)),
                ft.Card(  # Card para o gráfico
                    content=ft.Column([
                        container_filtros_grafico,  # Filtros acima do gráfico
                        ft.Container(controle_imagem_plot, padding=10, border_radius=10,
                                     bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE))  # Cor adaptável
                    ]),
                    elevation=3, expand=True
                )
            ], expand=3, spacing=15),  # Coluna esquerda ocupa mais espaço
            ft.Column([  # Coluna Direita: Ambiente e Qualidade do Ar
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([ft.Icon(ft.Icons.THERMOSTAT_ROUNDED, color=ft.Colors.WHITE, size=20),
                                    ft.Text(" Ambiente", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.WHITE)]),
                            texto_umidade, texto_temperatura
                        ], spacing=8),
                        bgcolor=ft.Colors.BLUE_GREY_600, padding=18, border_radius=10,
                    ), elevation=4
                ),
                ft.Card(content=container_qualidade_ar, elevation=4)  # Container da qualidade do ar (já estilizado)
            ], expand=1, spacing=15, run_spacing=15)  # Coluna direita ocupa menos espaço
        ], expand=True, vertical_alignment=ft.CrossAxisAlignment.START, spacing=15),
        ft.Row([  # Botões de Ação
            ft.FilledButton("Atualizar Dados", icon=ft.Icons.REFRESH_ROUNDED, on_click=ao_clicar_atualizar_dados,
                            height=45, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
            ft.FilledButton("Simular Novos Dados", icon=ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
                            on_click=ao_clicar_simular_dados, height=45,
                            style=ft.ButtonStyle(bgcolor=ft.Colors.TEAL_ACCENT_700,
                                                 shape=ft.RoundedRectangleBorder(radius=8)))
        ], alignment=ft.MainAxisAlignment.START, spacing=15, wrap=True)
        # wrap permite que botões quebrem linha em telas menores
    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True, spacing=18)  # Permite rolagem se conteúdo exceder

    # --- Construção do Conteúdo da Aba de Relatórios ---
    estilo_botao_relatorio = ft.ButtonStyle(  # Estilo para os botões de relatório (maiores e mais destacados)
        shape=ft.RoundedRectangleBorder(radius=10),
        padding=ft.padding.symmetric(horizontal=25, vertical=18),
        elevation=2,
        # bgcolor e color são herdados do tema do ElevatedButton, o que é bom para alternância de tema
        overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.PRIMARY)  # Efeito ao passar o mouse
    )
    # Painel com informações educacionais estáticas
    painel_info_educacional = ft.Card(
        elevation=2,
        content=ft.Container(
            ft.Column([
                ft.Row([ft.Icon(ft.Icons.SCHOOL_ROUNDED, color=ft.Colors.BLUE_700),
                        ft.Text("Saiba Mais", weight=ft.FontWeight.BOLD, size=18, color=ft.Colors.BLUE_700)]),
                ft.Text("Entenda os poluentes e seus impactos:", size=14, weight=ft.FontWeight.W_600),
                # W_600 para simular SemiBold
                ft.Text("CO2 (Dióxido de Carbono): Principal gás do efeito estufa, afeta o clima.", size=12,
                        italic=True),
                ft.Text("CH4 (Metano): Gás de efeito estufa potente, liberado por aterros e agricultura.", size=12,
                        italic=True),
                ft.Text("NH3 (Amônia): Irritante para vias aéreas, comum na agricultura e indústria.", size=12,
                        italic=True),
                ft.Text(
                    "VOCs (Compostos Orgânicos Voláteis): Contribuem para a formação de ozônio e problemas respiratórios.",
                    size=12, italic=True),
                ft.Text("A qualidade do ar 'Excelente' indica um ambiente seguro e saudável.", size=12,
                        weight=ft.FontWeight.W_600, color=ft.Colors.GREEN_700),
            ], spacing=8, alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.START),
            padding=15, border_radius=10
        )
    )

    conteudo_aba_relatorios = ft.Column([
        ft.Text("Relatórios e Diagnósticos do Sistema", weight=ft.FontWeight.BOLD, size=24, color=ft.Colors.PRIMARY),
        ft.Row([  # Botões para selecionar o tipo de relatório
            ft.ElevatedButton("Resumo da Análise", icon=ft.Icons.DESCRIPTION_ROUNDED,
                              on_click=lambda e: exibir_relatorio(page, "resumo"), style=estilo_botao_relatorio,
                              height=60),
            ft.ElevatedButton("Árvore de Decisão", icon=ft.Icons.ACCOUNT_TREE_ROUNDED,
                              on_click=lambda e: exibir_relatorio(page, "arvore"), disabled=not PIL_DISPONIVEL,
                              style=estilo_botao_relatorio, height=60),
            ft.ElevatedButton("Log do Sistema", icon=ft.Icons.LIST_ALT_ROUNDED,
                              on_click=lambda e: exibir_relatorio(page, "log"), style=estilo_botao_relatorio, height=60)
        ], spacing=15, alignment=ft.MainAxisAlignment.CENTER, wrap=True),  # wrap para responsividade
        ft.Row([  # Linha para dividir a área de relatório e o painel educacional
            ft.Column([  # Coluna para o conteúdo do relatório
                ft.Text("Conteúdo do Relatório:", size=14, weight=ft.FontWeight.W_600,
                        color=ft.Colors.ON_SURFACE_VARIANT),  # Cor adaptável ao tema
                ft.Card(
                    content=ft.Container(coluna_conteudo_relatorio, padding=15, border_radius=10, height=400,
                                         bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE)),
                    # Cor adaptável ao tema
                    elevation=2, expand=True
                )
            ], expand=2),  # Área de relatório ocupa mais espaço
            ft.Column([painel_info_educacional], expand=1, alignment=ft.MainAxisAlignment.START)  # Painel educacional
        ], expand=True, spacing=15, vertical_alignment=ft.CrossAxisAlignment.START),
    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True, spacing=18)

    # --- Configuração Final das Abas (Tabs) ---
    abas_principais = ft.Tabs(
        selected_index=0,  # Define a aba inicial
        animation_duration=300,  # Animação suave ao trocar de aba
        tabs=[
            ft.Tab(text="Monitoramento", icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                   content=conteudo_aba_monitoramento),
            ft.Tab(text="Relatórios", icon=ft.Icons.ARTICLE_OUTLINED, content=conteudo_aba_relatorios),
        ],
        expand=True,  # Ocupa todo o espaço horizontal possivel
        label_color=ft.Colors.PRIMARY,  # Cor da aba selecionada
        unselected_label_color=ft.Colors.ON_SURFACE_VARIANT  # Cor da aba não selecionada, adaptável ao tema
    )

    page.add(abas_principais)  # Adiciona o controle de abas à página
    atualizar_elementos_ui(page)  # Carrega os dados iniciais e atualiza a UI pela primeira vez


# Ponto de Entrada para Executar a Aplicação Flet
ft.app(target=main)
