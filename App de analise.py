import io
from datetime import datetime
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gestão de Obsoletos", layout="wide")

st.title("Gestão de Itens Obsoletos do Estoque")
st.caption("Análise de itens parados, última venda, custo imobilizado e plano de ação com priorização por grupo")

# =========================
# Configurações de sessão
# =========================
if 'campanha_data' not in st.session_state:
    st.session_state.campanha_data = {
        'itens_vendidos': [],
        'metas': {},
        'historico': []
    }

# =========================
# Funções auxiliares
# =========================
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {
        "Produto (item)": "Produto",
        "Produto": "Produto",
        "Unidade": "Unidade",
        "Grupo": "Grupo",
        "Marca": "Marca",
        "Quantidade": "Quantidade_Fisica",
        "Disponivel": "Quantidade",
        "Disponível": "Quantidade",
        "Custo Unit.": "Custo_Unit",
        "Custo Total": "Custo_Total",
        "Custo Total ": "Custo_Total",
        "Últ. Venda": "Ult_Venda",
        "Dias": "Dias",
        "Últ. Compra": "Ult_Compra",
        "Preço": "Preco",
    }
    df = df.rename(columns=mapa).copy()
    
    # Se não encontrou a coluna "Quantidade" (Disponível), tenta criar a partir da coluna "Quantidade_Fisica"
    if "Quantidade" not in df.columns and "Quantidade_Fisica" in df.columns:
        df["Quantidade"] = df["Quantidade_Fisica"]
        st.warning("⚠️ Coluna 'Disponível' não encontrada. Utilizando a coluna 'Quantidade' como base.")
    
    return df


def validar_colunas(df: pd.DataFrame):
    obrigatorias = [
        "Produto",
        "Unidade",
        "Grupo",
        "Marca",
        "Quantidade",
        "Custo_Unit",
        "Custo_Total",
        "Ult_Venda",
        "Dias",
        "Ult_Compra",
        "Preco",
    ]
    faltando = [c for c in obrigatorias if c not in df.columns]
    
    if "Quantidade" in faltando:
        st.error("❌ Coluna 'Disponível' ou 'Disponivel' não encontrada na planilha.")
        st.info("Verifique se sua planilha tem uma coluna chamada 'Disponível' ou 'Disponivel' (sem acento) indicando a quantidade disponível para venda.")
    
    return faltando


def converter_para_float(valor):
    """Converte valores monetários de string para float de forma segura"""
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    
    valor_str = str(valor).strip()
    valor_str = valor_str.replace('R$', '').replace('r$', '').strip()
    
    if ',' in valor_str:
        if '.' in valor_str:
            partes = valor_str.split(',')
            parte_inteira = partes[0].replace('.', '')
            valor_convertido = f"{parte_inteira}.{partes[1]}"
        else:
            valor_convertido = valor_str.replace(',', '.')
    else:
        valor_convertido = valor_str
    
    try:
        return float(valor_convertido)
    except:
        return np.nan


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Datas
    df["Ult_Venda"] = pd.to_datetime(df["Ult_Venda"], format="%d/%m/%Y", errors="coerce")
    df["Ult_Compra"] = pd.to_datetime(df["Ult_Compra"], format="%d/%m/%Y", errors="coerce")
    
    # Calcular dias sem compra
    data_atual = datetime.now()
    df["Dias_Sem_Compra"] = (data_atual - df["Ult_Compra"]).dt.days
    df["Dias_Sem_Compra"] = df["Dias_Sem_Compra"].fillna(0).astype(int)
    
    # Converter colunas numéricas
    # Quantidade Disponível - principal coluna
    df["Quantidade"] = df["Quantidade"].apply(converter_para_float)
    df["Quantidade"] = df["Quantidade"].fillna(0).astype(int)
    
    # Se existir Quantidade_Fisica (antiga coluna Quantidade), manter para referência
    if "Quantidade_Fisica" in df.columns:
        df["Quantidade_Fisica"] = df["Quantidade_Fisica"].apply(converter_para_float)
        df["Quantidade_Fisica"] = df["Quantidade_Fisica"].fillna(0).astype(int)
    
    # Custo Unitário
    df["Custo_Unit"] = df["Custo_Unit"].apply(converter_para_float)
    df["Custo_Unit"] = df["Custo_Unit"].fillna(0)
    
    # Custo Total - recalcular baseado na Quantidade Disponível
    df["Custo_Total"] = df["Custo_Total"].apply(converter_para_float)
    df["Custo_Total"] = df["Custo_Total"].fillna(0)
    
    # Recalcular custo total baseado na Quantidade Disponível
    custo_calculado = df["Quantidade"] * df["Custo_Unit"]
    mask_custo_invalido = (df["Custo_Total"] == 0) | (abs(df["Custo_Total"] - custo_calculado) > 0.01)
    if mask_custo_invalido.any():
        df.loc[mask_custo_invalido, "Custo_Total"] = custo_calculado[mask_custo_invalido]
    
    # Dias
    df["Dias"] = df["Dias"].apply(converter_para_float)
    df["Dias"] = df["Dias"].fillna(0).astype(int)
    
    # Preço
    df["Preco"] = df["Preco"].apply(converter_para_float)
    df["Preco"] = df["Preco"].fillna(0)
    
    # Margem bruta estimada
    df["Margem_R$"] = df["Preco"] - df["Custo_Unit"]
    df["Margem_%"] = ((df["Preco"] - df["Custo_Unit"]) / df["Preco"]) * 100
    df["Margem_%"] = df["Margem_%"].replace([np.inf, -np.inf], np.nan)
    df["Margem_%"] = df["Margem_%"].round(2)
    df["Margem_%"] = df["Margem_%"].fillna(0)
    
    # Classificação gerencial
    def classificar_obsolescencia(dias):
        if pd.isna(dias):
            return "Sem informação"
        if dias >= 720:
            return "Crítico"
        if dias >= 540:
            return "Alto"
        if dias >= 360:
            return "Obsoleto"
        if dias >= 180:
            return "Atenção"
        return "Saudável"
    
    df["Faixa_Obsolescencia"] = df["Dias"].apply(classificar_obsolescencia)
    
    def sugerir_acao(row):
        dias = row["Dias"]
        margem = row["Margem_%"]
        qtd = row["Quantidade"]
        if pd.isna(dias):
            return "Revisar cadastro e histórico"
        if dias >= 720:
            return "Liquidação forte / devolução / sucatear / kit"
        if dias >= 540:
            return "Campanha comercial agressiva"
        if dias >= 360 and margem is not None and pd.notna(margem) and margem > 20:
            return "Promover com desconto controlado"
        if dias >= 360 and qtd is not None and pd.notna(qtd) and qtd <= 3:
            return "Venda direcionada item a item"
        if dias >= 360:
            return "Montar plano de giro por grupo"
        return "Manter acompanhamento"
    
    df["Acao_Sugerida"] = df.apply(sugerir_acao, axis=1)
    
    # Calcular métricas por grupo para priorização
    df = calcular_priorizacao_grupo(df)
    
    return df


def calcular_priorizacao_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula percentuais de participação por grupo para priorização"""
    df = df.copy()
    
    # Totais por grupo baseado na Quantidade Disponível
    grupo_totais = df.groupby('Grupo').agg({
        'Quantidade': 'sum',
        'Custo_Total': 'sum'
    }).rename(columns={
        'Quantidade': 'Total_Qtd_Grupo',
        'Custo_Total': 'Total_Valor_Grupo'
    })
    
    # Merge com os totais
    df = df.merge(grupo_totais, on='Grupo', how='left')
    
    # Calcular percentuais de participação dentro do grupo
    df['%_Qtd_no_Grupo'] = (df['Quantidade'] / df['Total_Qtd_Grupo'] * 100).round(2)
    df['%_Valor_no_Grupo'] = (df['Custo_Total'] / df['Total_Valor_Grupo'] * 100).round(2)
    
    # Substituir NaN por 0
    df['%_Qtd_no_Grupo'] = df['%_Qtd_no_Grupo'].fillna(0)
    df['%_Valor_no_Grupo'] = df['%_Valor_no_Grupo'].fillna(0)
    
    # Calcular participação acumulada para análise ABC
    df = df.sort_values(['Grupo', 'Custo_Total'], ascending=[True, False])
    df['%_Valor_Acumulado_Grupo'] = df.groupby('Grupo')['%_Valor_no_Grupo'].cumsum()
    df['%_Valor_Acumulado_Grupo'] = df['%_Valor_Acumulado_Grupo'].fillna(0)
    
    # Classificação ABC dentro do grupo
    def classificar_abc(perc_acumulado):
        if pd.isna(perc_acumulado):
            return 'C (10% do valor)'
        elif perc_acumulado <= 70:
            return 'A (70% do valor)'
        elif perc_acumulado <= 90:
            return 'B (20% do valor)'
        else:
            return 'C (10% do valor)'
    
    df['Classificacao_ABC'] = df['%_Valor_Acumulado_Grupo'].apply(classificar_abc)
    
    # Score de priorização combinado
    max_dias = df['Dias'].max()
    if max_dias > 0:
        df['Score_Tempo'] = (df['Dias'] / max_dias * 100).round(2)
    else:
        df['Score_Tempo'] = 0
    
    df['Score_Tempo'] = df['Score_Tempo'].fillna(0)
    df['%_Valor_no_Grupo'] = df['%_Valor_no_Grupo'].fillna(0)
    
    df['Score_Priorizacao'] = (df['%_Valor_no_Grupo'] * 0.6 + df['Score_Tempo'] * 0.4).round(2)
    df['Score_Priorizacao'] = df['Score_Priorizacao'].fillna(0)
    
    # Ranking dentro do grupo - tratar NaN antes do rank
    df['Rank_Grupo'] = df.groupby('Grupo')['Score_Priorizacao'].rank(method='dense', ascending=False)
    df['Rank_Grupo'] = df['Rank_Grupo'].fillna(999)
    df['Rank_Grupo'] = df['Rank_Grupo'].astype(int)
    
    return df


def formatar_moeda(valor):
    """Formata valores monetários no padrão brasileiro"""
    if pd.isna(valor) or valor == 0:
        return "R$ 0,00"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_numero(valor):
    """Formata números grandes com separador de milhar"""
    if pd.isna(valor):
        return "-"
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor):
    """Formata percentuais"""
    if pd.isna(valor):
        return "-"
    return f"{valor:.1f}%"


def filtrar_por_percentual(data, valores, percentual_minimo, nome_agrupamento="Outros"):
    """Filtra dados agrupando itens com percentual menor que o mínimo como 'Outros'"""
    if len(data) == 0:
        return data
    
    total = sum(valores)
    percentuais = [(val / total * 100) if total > 0 else 0 for val in valores]
    
    dados_filtrados = []
    for nome, valor, perc in zip(data, valores, percentuais):
        if perc >= percentual_minimo:
            dados_filtrados.append((nome, valor, perc))
    
    outros_valor = sum([valor for nome, valor, perc in zip(data, valores, percentuais) if perc < percentual_minimo])
    outros_perc = (outros_valor / total * 100) if total > 0 else 0
    
    if outros_valor > 0:
        dados_filtrados.append((nome_agrupamento, outros_valor, outros_perc))
    
    return dados_filtrados


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Configurações")

# Slider para dias sem venda (obsoleto)
st.sidebar.subheader("📅 Filtro por Dias sem Venda")
limite_obsoleto = st.sidebar.slider(
    "Dias sem venda para considerar obsoleto", 
    min_value=30, 
    max_value=720, 
    value=360, 
    step=30,
    help="Itens com mais dias sem venda que este valor serão considerados obsoletos"
)

# Novo slider para dias sem compra
st.sidebar.subheader("📅 Filtro por Dias sem Compra")
limite_sem_compra = st.sidebar.slider(
    "Dias sem compra", 
    min_value=0, 
    max_value=720, 
    value=0, 
    step=30,
    help="Filtrar itens com mais de X dias sem compra (0 = sem filtro)"
)

# Slider para top itens
st.sidebar.subheader("📊 Configuração de Visualização")
limite_top = st.sidebar.slider("Top itens por custo imobilizado", 5, 100, 20)

# Ordenação
ordem_priorizacao = st.sidebar.radio(
    "Ordenar por:",
    ["Score de Priorização (Recomendado)", "Valor no Grupo (%)", "Tempo Parado (Dias)", "Rank no Grupo"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Configuração de Gráficos")
percentual_minimo_grafico = st.sidebar.slider(
    "Percentual mínimo para exibir nos gráficos (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.5,
    step=0.5,
    help="Itens ou grupos com participação menor que este percentual serão agrupados como 'Outros' nos gráficos"
)

arquivo = st.sidebar.file_uploader(
    "📁 Envie sua planilha de estoque (.xlsx ou .csv)",
    type=["xlsx", "csv"],
)

modelo_df = pd.DataFrame(
    {
        "Produto (item)": ["ROLAMENTO 6205", "MOTOR ELÉTRICO 5CV", "CORREIA BANDEX"],
        "Unidade": ["PC", "PC", "M"],
        "Grupo": ["Rolamentos", "Motores", "Correias"],
        "Marca": ["SKF", "WEG", "GATES"],
        "Quantidade": [10, 0, 45],
        "Disponivel": [8, 0, 40],
        "Custo Unit.": ["42,50", "1.250,00", "18,90"],
        "Custo Total": ["340,00", "0,00", "756,00"],
        "Últ. Venda": ["10/01/2025", "15/12/2024", "20/01/2025"],
        "Dias": [430, 280, 410],
        "Últ. Compra": ["08/12/2024", "10/10/2024", "05/01/2025"],
        "Preço": ["69,90", "1.890,00", "32,50"],
    }
)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    modelo_df.to_excel(writer, index=False, sheet_name="Estoque")
buffer.seek(0)

st.sidebar.download_button(
    label="📥 Baixar modelo de planilha",
    data=buffer,
    file_name="modelo_estoque_obsoletos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if arquivo is None:
    st.info("👈 Envie uma planilha para iniciar a análise.")
    
    st.markdown("""
    ### 📋 Formato esperado da planilha:
    
    | Coluna | Descrição |
    |--------|-----------|
    | **Produto (item)** | Nome do produto |
    | **Unidade** | Unidade de medida (PC, KG, etc) |
    | **Grupo** | Família do produto |
    | **Marca** | Marca do fabricante |
    | **Quantidade** | Saldo físico total |
    | **Disponivel** | Quantidade disponível para venda (principal) |
    | **Custo Unit.** | Custo unitário |
    | **Custo Total** | Custo total |
    | **Últ. Venda** | Data da última venda (DD/MM/AAAA) |
    | **Dias** | Dias sem venda |
    | **Últ. Compra** | Data da última compra (DD/MM/AAAA) |
    | **Preço** | Preço de venda |
    
    > ⚠️ **Importante**: O app utiliza a coluna **"Disponivel"** como base para todos os cálculos de quantidade e custo total.
    """)
    st.stop()

# =========================
# Leitura do arquivo
# =========================
try:
    if arquivo.name.endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)
        
    with st.expander("📋 Ver colunas encontradas no arquivo"):
        st.write("Colunas detectadas:", list(df.columns))
        
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

# =========================
# Preparação dos dados
# =========================
df = normalizar_colunas(df)
faltando = validar_colunas(df)
if faltando:
    st.error(f"❌ A planilha não possui as colunas obrigatórias: {', '.join(faltando)}")
    st.stop()

df = tratar_dados(df)

# =========================
# Filtros
# =========================
st.subheader("🔍 Filtros")
colf1, colf2, colf3 = st.columns(3)

with colf1:
    grupos = ["Todos"] + sorted([g for g in df["Grupo"].dropna().astype(str).unique()])
    grupo_sel = st.selectbox("Grupo", grupos)

with colf2:
    marcas = ["Todas"] + sorted([m for m in df["Marca"].dropna().astype(str).unique()])
    marca_sel = st.selectbox("Marca", marcas)

with colf3:
    termo = st.text_input("Buscar produto")

filtrado = df.copy()
if grupo_sel != "Todos":
    filtrado = filtrado[filtrado["Grupo"].astype(str) == grupo_sel]
if marca_sel != "Todas":
    filtrado = filtrado[filtrado["Marca"].astype(str) == marca_sel]
if termo:
    filtrado = filtrado[filtrado["Produto"].astype(str).str.contains(termo, case=False, na=False)]

# Calcular dias sem compra (já deve existir no DataFrame)
if "Dias_Sem_Compra" not in filtrado.columns:
    data_atual = datetime.now()
    filtrado["Dias_Sem_Compra"] = (data_atual - filtrado["Ult_Compra"]).dt.days

# Aplicar filtro de dias sem compra
if limite_sem_compra > 0:
    filtrado = filtrado[filtrado["Dias_Sem_Compra"] >= limite_sem_compra]

# Aplicar filtro de obsoletos (dias sem venda)
obsoletos = filtrado[(filtrado["Dias"] >= limite_obsoleto) & (filtrado["Quantidade"] > 0)].copy()

# Exibir informações dos filtros aplicados
st.info(f"📌 Filtros aplicados: {len(filtrado)} itens totais | {len(obsoletos)} obsoletos | Dias sem compra ≥ {limite_sem_compra if limite_sem_compra > 0 else 'todos'}")

# =========================
# Métricas executivas
# =========================
st.subheader("📊 Visão executiva")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Itens no estoque", formatar_numero(len(filtrado)))
col2.metric("Itens obsoletos", formatar_numero(len(obsoletos)))
col3.metric("Custo obsoleto", formatar_moeda(obsoletos["Custo_Total"].sum()))

total_custo = filtrado["Custo_Total"].sum()
participacao = 0
if total_custo > 0:
    participacao = (obsoletos["Custo_Total"].sum() / total_custo) * 100
col4.metric("% custo imobilizado", f"{participacao:.1f}%")

st.caption(f"✅ Todos os cálculos utilizam a coluna **'Disponivel'** como base para quantidades e custos")

# =========================
# Estratégia sugerida
# =========================
with st.expander("📌 Estratégia e metas recomendadas", expanded=True):
    st.markdown(
        f"""
### Objetivo central
Reduzir o valor imobilizado em itens sem giro e aumentar a conversão do estoque parado em caixa.

### Metas sugeridas
- Reduzir em **20%** o custo total dos itens com mais de **{limite_obsoleto} dias** sem venda em até **90 dias**
- Reduzir em **35%** em até **180 dias**
- Impedir novas compras de itens classificados como **Obsoleto, Alto ou Crítico**, salvo justificativa formal
- Criar rotina de revisão quinzenal dos **20 maiores itens obsoletos por custo total**
- Definir plano por grupo: **promoção, devolução, kit, venda técnica, transferência ou baixa**

### Regras de gestão
1. Itens com **360 a 539 dias**: campanha comercial e venda assistida
2. Itens com **540 a 719 dias**: desconto controlado, combo, oferta ativa
3. Itens com **720+ dias**: liquidar, devolver ao fornecedor, transformar em kit, reaproveitar ou baixar
4. Toda compra de item sem venda acima de 180 dias deve exigir aprovação
5. Acompanhar mensalmente: valor obsoleto, giro recuperado e itens resolvidos
        """
    )

# =========================
# Abas principais
# =========================
aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
    "📊 Análise por Grupo",
    "🎯 Itens Priorizados",
    "⚠️ Itens > 360 dias",
    "📈 Top Custo Imobilizado",
    "📋 Plano de Ação",
    "📁 Base Completa",
])

# =========================
# Aba 1: Análise por Grupo
# =========================
with aba1:
    st.subheader("Análise de Priorização por Grupo")
    
    if len(obsoletos) > 0:
        resumo_grupo = obsoletos.groupby('Grupo').agg({
            'Produto': 'count',
            'Quantidade': 'sum',
            'Custo_Total': 'sum',
            'Dias': 'mean'
        }).round(2)
        resumo_grupo.columns = ['Qtd Itens', 'Qtd Disponível', 'Valor Total (R$)', 'Média Dias Parado']
        resumo_grupo = resumo_grupo.sort_values('Valor Total (R$)', ascending=False)
        
        total_valor = resumo_grupo['Valor Total (R$)'].sum()
        resumo_grupo['% do Total'] = (resumo_grupo['Valor Total (R$)'] / total_valor * 100).round(1)
        resumo_grupo['Valor Total (R$)'] = resumo_grupo['Valor Total (R$)'].apply(formatar_moeda)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Impacto por Grupo")
            st.dataframe(resumo_grupo, use_container_width=True)
        
        with col2:
            grupo_valores = obsoletos.groupby('Grupo')['Custo_Total'].sum().reset_index()
            grupo_valores.columns = ['Grupo', 'Valor']
            
            dados_filtrados = filtrar_por_percentual(
                grupo_valores['Grupo'].tolist(),
                grupo_valores['Valor'].tolist(),
                percentual_minimo_grafico,
                "Outros Grupos"
            )
            
            if dados_filtrados:
                grupos_filtrados = [item[0] for item in dados_filtrados]
                valores_filtrados = [item[1] for item in dados_filtrados]
                
                fig = px.pie(
                    values=valores_filtrados,
                    names=grupos_filtrados,
                    title=f"Distribuição do Valor em Obsoletos por Grupo<br><sup>Grupos com > {percentual_minimo_grafico}% de participação</sup>",
                    hole=0.3
                )
                fig.update_traces(textinfo='label+percent', textposition='inside')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 🎯 Priorização por Grupo (Valor vs Quantidade)")
        comparativo = obsoletos.groupby('Grupo').agg({
            'Custo_Total': 'sum',
            'Produto': 'count'
        }).round(2)
        
        total_valor_comparativo = comparativo['Custo_Total'].sum()
        comparativo['%_Participacao'] = (comparativo['Custo_Total'] / total_valor_comparativo * 100)
        grupos_principais = comparativo[comparativo['%_Participacao'] >= percentual_minimo_grafico].index.tolist()
        
        if len(grupos_principais) > 0:
            comparativo_filtrado = comparativo.loc[grupos_principais]
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Valor em Obsoletos (R$)', 'Quantidade de Itens Obsoletos'))
            fig.add_trace(go.Bar(x=comparativo_filtrado.index, y=comparativo_filtrado['Custo_Total'], name='Valor', marker_color='indianred'), row=1, col=1)
            fig.add_trace(go.Bar(x=comparativo_filtrado.index, y=comparativo_filtrado['Produto'], name='Quantidade', marker_color='lightblue'), row=1, col=2)
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 Não há itens obsoletos disponíveis para venda!")

# =========================
# Aba 2: Itens Priorizados
# =========================
with aba2:
    st.subheader("🎯 Itens Obsoletos Priorizados por Grupo")
    st.markdown("**Critério de priorização:** 60% representatividade no valor do grupo + 40% tempo parado")
    
    if len(obsoletos) > 0:
        if ordem_priorizacao == "Score de Priorização (Recomendado)":
            priorizados = obsoletos.sort_values('Score_Priorizacao', ascending=False)
        elif ordem_priorizacao == "Valor no Grupo (%)":
            priorizados = obsoletos.sort_values('%_Valor_no_Grupo', ascending=False)
        elif ordem_priorizacao == "Tempo Parado (Dias)":
            priorizados = obsoletos.sort_values('Dias', ascending=False)
        else:
            priorizados = obsoletos.sort_values(['Grupo', 'Rank_Grupo'])
        
        colunas_priorizacao = [
            "Rank_Grupo", "Produto", "Grupo", "Marca", "Quantidade", 
            "Custo_Total", "%_Valor_no_Grupo", "Dias", "Ult_Compra", "Ult_Venda",
            "Faixa_Obsolescencia", "Score_Priorizacao", "Classificacao_ABC", "Acao_Sugerida"
        ]
        
        priorizados_display = priorizados[colunas_priorizacao].copy()
        priorizados_display["Custo_Total"] = priorizados_display["Custo_Total"].apply(formatar_moeda)
        priorizados_display["%_Valor_no_Grupo"] = priorizados_display["%_Valor_no_Grupo"].apply(formatar_percentual)
        priorizados_display["Score_Priorizacao"] = priorizados_display["Score_Priorizacao"].apply(formatar_percentual)
        priorizados_display["Ult_Compra"] = priorizados_display["Ult_Compra"].dt.strftime('%d/%m/%Y')
        priorizados_display["Ult_Venda"] = priorizados_display["Ult_Venda"].dt.strftime('%d/%m/%Y')
        
        def highlight_prioridade(row):
            if row['Rank_Grupo'] <= 3:
                return ['background-color: #ffcccc'] * len(row)
            elif row['Rank_Grupo'] <= 10:
                return ['background-color: #ffffcc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(priorizados_display.style.apply(highlight_prioridade, axis=1), use_container_width=True)
        
        st.markdown("#### 📊 Top 20 Itens Mais Prioritários")
        top_20 = priorizados.head(20)
        if len(top_20) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_20['Produto'],
                y=top_20['Score_Priorizacao'],
                text=top_20['Score_Priorizacao'].round(1),
                textposition='auto',
                marker_color=top_20['Score_Priorizacao'],
                marker_colorscale='Reds'
            ))
            fig.update_layout(title="Score de Priorização por Item", xaxis_title="Produto", yaxis_title="Score (%)", height=500, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            "📥 Baixar relatório de itens priorizados",
            data=priorizados[colunas_priorizacao].to_csv(index=False, sep=';', decimal=',').encode("utf-8-sig"),
            file_name="itens_priorizados_obsoletos.csv",
            mime="text/csv",
        )
    else:
        st.success("🎉 Não há itens obsoletos para priorizar!")

# =========================
# Aba 3: Itens > 360 dias
# =========================
with aba3:
    st.markdown("### Itens disponíveis há mais de 360 dias sem venda")
    
    if len(obsoletos) > 0:
        visao1 = obsoletos[[
            "Produto", "Unidade", "Grupo", "Marca", "Quantidade", "Custo_Unit", "Custo_Total",
            "Ult_Compra", "Dias", "Faixa_Obsolescencia", "%_Valor_no_Grupo", "Rank_Grupo", "Acao_Sugerida"
        ]].sort_values(by=["Dias", "Custo_Total"], ascending=[False, False])
        
        visao1_display = visao1.copy()
        visao1_display["Custo_Unit"] = visao1_display["Custo_Unit"].apply(formatar_moeda)
        visao1_display["Custo_Total"] = visao1_display["Custo_Total"].apply(formatar_moeda)
        visao1_display["%_Valor_no_Grupo"] = visao1_display["%_Valor_no_Grupo"].apply(formatar_percentual)
        visao1_display["Ult_Compra"] = visao1_display["Ult_Compra"].dt.strftime('%d/%m/%Y')
        
        st.dataframe(visao1_display, use_container_width=True)
        
        st.download_button(
            "📥 Baixar visão de obsoletos",
            data=visao1.to_csv(index=False, sep=';', decimal=',').encode("utf-8-sig"),
            file_name="itens_obsoletos.csv",
            mime="text/csv",
        )
    else:
        st.success("🎉 Não há itens obsoletos disponíveis para venda com os filtros atuais!")

# =========================
# Aba 4: Top Custo Imobilizado
# =========================
with aba4:
    st.markdown("### Top itens obsoletos por custo total")
    
    if len(obsoletos) > 0:
        top_custo = obsoletos.sort_values(by="Custo_Total", ascending=False).head(limite_top)
        
        if len(top_custo) > 0:
            fig = px.bar(
                top_custo,
                x='Produto',
                y='Custo_Total',
                color='Grupo',
                title=f"Top {limite_top} Itens com Maior Custo em Obsoletos",
                text_auto='.2s'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            top_custo_display = top_custo[["Produto", "Grupo", "Marca", "Quantidade", "Custo_Total", "Dias", "%_Valor_no_Grupo", "Ult_Venda", "Acao_Sugerida"]].copy()
            top_custo_display["Custo_Total"] = top_custo_display["Custo_Total"].apply(formatar_moeda)
            top_custo_display["%_Valor_no_Grupo"] = top_custo_display["%_Valor_no_Grupo"].apply(formatar_percentual)
            top_custo_display["Ult_Venda"] = top_custo_display["Ult_Venda"].dt.strftime('%d/%m/%Y')
            st.dataframe(top_custo_display, use_container_width=True)
    else:
        st.info("Nenhum item obsoleto encontrado com os filtros atuais.")

# =========================
# Aba 5: Plano de Ação
# =========================
with aba5:
    st.markdown("### Plano de ação gerado automaticamente com priorização")
    
    if len(obsoletos) > 0:
        plano = obsoletos[["Produto", "Grupo", "Marca", "Quantidade", "Custo_Total", 
                           "%_Valor_no_Grupo", "Rank_Grupo", "Dias", "Ult_Venda", 
                           "Faixa_Obsolescencia", "Acao_Sugerida"]].copy()
        
        plano["Responsavel"] = plano["Faixa_Obsolescencia"].map({
            "Crítico": "Diretoria + Comercial",
            "Alto": "Gerente Comercial",
            "Obsoleto": "Coordenador Comercial",
            "Atenção": "Analista de Estoque",
        }).fillna("Equipe de Estoque")
        
        plano["Prazo"] = plano["Faixa_Obsolescencia"].map({
            "Crítico": "7 dias",
            "Alto": "15 dias",
            "Obsoleto": "30 dias",
            "Atenção": "60 dias",
        }).fillna("Monitorar")
        
        plano["Prioridade"] = plano["Rank_Grupo"].apply(lambda x: "Alta" if x <= 3 else "Média" if x <= 10 else "Baixa")
        plano["Status"] = "Pendente"
        plano = plano.sort_values(['Prioridade', 'Rank_Grupo'])
        
        plano_display = plano.copy()
        plano_display["Custo_Total"] = plano_display["Custo_Total"].apply(formatar_moeda)
        plano_display["%_Valor_no_Grupo"] = plano_display["%_Valor_no_Grupo"].apply(formatar_percentual)
        plano_display["Ult_Venda"] = plano_display["Ult_Venda"].dt.strftime('%d/%m/%Y')
        
        st.dataframe(plano_display, use_container_width=True)
        
        st.download_button(
            "📥 Baixar plano de ação",
            data=plano.to_csv(index=False, sep=';', decimal=',').encode("utf-8-sig"),
            file_name="plano_acao_obsoletos.csv",
            mime="text/csv",
        )
    else:
        st.info("Nenhum item obsoleto para gerar plano de ação.")

# =========================
# Aba 6: Base Completa
# =========================
with aba6:
    st.markdown("### Base completa tratada com métricas de priorização")
    st.caption(f"📌 **Quantidade** = Disponível para venda | Total de itens: {len(filtrado)}")
    
    base_display = filtrado.copy()
    
    for col in ['Custo_Unit', 'Custo_Total', 'Preco', 'Margem_R$']:
        if col in base_display.columns:
            base_display[col] = base_display[col].apply(formatar_moeda)
    
    for col in ['%_Qtd_no_Grupo', '%_Valor_no_Grupo', 'Score_Priorizacao']:
        if col in base_display.columns:
            base_display[col] = base_display[col].apply(formatar_percentual)
    
    if 'Ult_Venda' in base_display.columns:
        base_display['Ult_Venda'] = base_display['Ult_Venda'].dt.strftime('%d/%m/%Y')
    if 'Ult_Compra' in base_display.columns:
        base_display['Ult_Compra'] = base_display['Ult_Compra'].dt.strftime('%d/%m/%Y')
    if 'Margem_%' in base_display.columns:
        base_display['Margem_%'] = base_display['Margem_%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    
    st.dataframe(base_display, use_container_width=True)
    
    st.download_button(
        "📥 Baixar base completa",
        data=filtrado.to_csv(index=False, sep=';', decimal=',').encode("utf-8-sig"),
        file_name="base_completa_estoque.csv",
        mime="text/csv",
    )

# =========================
# Rodapé
# =========================
st.divider()
st.markdown("📊 **Dashboard de Gestão de Obsoletos** | Baseado na quantidade **Disponivel** para venda")