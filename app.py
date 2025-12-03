import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv

# --- IMPORTAÇÕES DE IA ---
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
st.set_page_config(page_title="Vuca Smart", layout="wide")

# --- FUNÇÕES DE DADOS ---
@st.cache_data
def processar_nova_ficha(arquivo):
    try:
        df = pd.read_csv(arquivo, sep=';', encoding='latin1') if isinstance(arquivo, str) else pd.read_csv(arquivo, sep=';', encoding='latin1')
        df.columns = df.columns.str.replace('"', '').str.strip().str.lower()
        
        if 'valor_custo' not in df.columns and 'valor custo' not in df.columns: return pd.DataFrame()

        mapa = {'produto_principal': 'produto_nome', 'valor_custo': 'custo_componente'}
        df = df.rename(columns=mapa)
        df['custo_componente'] = pd.to_numeric(df['custo_componente'], errors='coerce')
        df = df.dropna(subset=['custo_componente'])
        
        if 'produto_nome' in df.columns:
            df['produto_nome'] = df['produto_nome'].astype(str).str.strip().str.replace(' +', ' ', regex=True).str.upper()
            return df.groupby('produto_nome')['custo_componente'].sum().reset_index().rename(columns={'custo_componente': 'custo_producao'})
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data
def filtrar_vendas(arquivo):
    try:
        df = pd.read_csv(arquivo, sep=';', encoding='latin1') if isinstance(arquivo, str) else pd.read_csv(arquivo, sep=';', encoding='latin1')
        df.columns = df.columns.str.replace('"', '').str.strip().str.upper()
        
        if 'PRODUTO DE VENDA' not in df.columns: return pd.DataFrame()
        if 'UNIDADE' in df.columns: df.drop(['UNIDADE'], axis=1, inplace=True)
            
        df = df.rename(columns={
            'PRODUTO DE VENDA': 'produto_nome', 'VENDA DE FRENTE DE LOJA': 'vendas_loja',
            'VENDA DELIVERY': 'vendas_delivery', 'RECEITA FRENTE DE LOJA': 'receita_loja',
            'RECEITA DELIVERY': 'receita_delivery'
        })
        
        df['produto_nome'] = df['produto_nome'].astype(str).str.strip().str.replace(' +', ' ', regex=True).str.upper()
        
        for col in ['vendas_loja', 'vendas_delivery', 'receita_loja', 'receita_delivery']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        df = df.fillna(0)
        df['popularidade'] = df['vendas_loja'] + df['vendas_delivery']
        df['receita_total'] = df['receita_loja'] + df['receita_delivery']
        df['preco_venda'] = np.where(df['popularidade'] > 0, df['receita_total'] / df['popularidade'], 0)
        return df[['produto_nome', 'popularidade', 'preco_venda', 'receita_total']]
    except: return pd.DataFrame()

def classificar_produto(row, pop, luc):
    if row['popularidade'] >= pop and row['lucratividade'] >= luc: return '⭐ Estrela'
    elif row['popularidade'] >= pop and row['lucratividade'] < luc: return '🐴 Burro de Carga'
    elif row['popularidade'] < pop and row['lucratividade'] >= luc: return '🧩 Quebra-cabeça'
    else: return '🐶 Cão'

def limpar_texto_ia(texto_obj):
    texto = str(texto_obj.raw) if hasattr(texto_obj, 'raw') else str(texto_obj)
    return texto.replace("$", "\\$")

# --- FUNÇÃO DO AGENTE ---
def executar_agente(dados_csv, provedor, modelo, api_key):
    # Configuração do LLM
    if provedor == "Gemini":
        llm = ChatGoogleGenerativeAI(
            model=modelo.split("/")[-1],
            verbose=True, temperature=0.5, google_api_key=api_key
        )
    else:
        # Para DeepSeek, Perplexity e ChatGPT usamos o LLM nativo via LiteLLM
        llm = LLM(model=modelo, api_key=api_key)

    analista = Agent(
        role="Analista de Menu",
        goal="Identificar itens críticos (Estrelas, Cães, etc) e oportunidades de lucro.",
        backstory="Especialista em Engenharia de Cardápio.",
        verbose=True, llm=llm, allow_delegation=False
    )
    
    consultor = Agent(
        role="Consultor Estratégico",
        goal="Criar um plano de ação prático para o dono do restaurante.",
        backstory="Consultor experiente que dá dicas diretas e acionáveis.",
        verbose=True, llm=llm, allow_delegation=False
    )
    
    t1 = Task(
        description=f"Analise estes dados de cardápio:\n{dados_csv}\nIdentifique: 1. O item mais lucrativo ('Estrela' ou 'Quebra-cabeça'). 2. Um item que dá prejuízo ou lucro baixo ('Cão' ou 'Burro').",
        expected_output="Resumo técnico dos itens selecionados.", agent=analista
    )
    
    t2 = Task(
        description="Escreva 3 recomendações práticas e curtas para o dono do restaurante com base na análise.",
        expected_output="Texto formatado com as dicas.", agent=consultor, context=[t1]
    )
    
    crew = Crew(agents=[analista, consultor], tasks=[t1, t2], process=Process.sequential)
    return crew.kickoff()

# --- INTERFACE ---
st.sidebar.title("🔧 Configurações da IA")
provedor = st.sidebar.selectbox("Escolha a Inteligência:", ["Gemini", "DeepSeek", "Perplexity", "ChatGPT"])

api_key_final = None
modelo_selecionado = None

# Configuração Dinâmica de Provedores
if provedor == "Gemini":
    mod = st.sidebar.selectbox("Modelo:", ["gemini-1.5-flash", "gemini-pro"])
    modelo_selecionado = f"google_gemini/{mod}"
    api_key_final = os.getenv("GOOGLE_API_KEY") or st.sidebar.text_input("Google API Key:", type="password")
    if api_key_final: os.environ["GOOGLE_API_KEY"] = api_key_final

elif provedor == "DeepSeek":
    mod = st.sidebar.selectbox("Modelo:", ["deepseek-chat", "deepseek-coder"])
    modelo_selecionado = f"deepseek/{mod}"
    api_key_final = os.getenv("DEEPSEEK_API_KEY") or st.sidebar.text_input("DeepSeek API Key:", type="password")
    if api_key_final: os.environ["DEEPSEEK_API_KEY"] = api_key_final

elif provedor == "Perplexity":
    # --- CORREÇÃO: NOMES ATUALIZADOS DOS MODELOS SONAR ---
    mod = st.sidebar.selectbox("Modelo:", ["sonar-pro", "sonar", "sonar-reasoning"])
    modelo_selecionado = f"perplexity/{mod}"
    api_key_final = os.getenv("PERPLEXITY_API_KEY") or st.sidebar.text_input("Perplexity API Key:", type="password")
    if api_key_final: os.environ["PERPLEXITY_API_KEY"] = api_key_final

elif provedor == "ChatGPT":
    mod = st.sidebar.selectbox("Modelo:", ["gpt-4o-mini", "gpt-4o"])
    modelo_selecionado = f"openai/{mod}"
    api_key_final = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key:", type="password")
    if api_key_final: os.environ["OPENAI_API_KEY"] = api_key_final

col1, col2 = st.columns([1, 17]) 
with col1:
    if os.path.exists("dataset/logovuca.png"): st.image("dataset/logovuca.png", width=80)
    else: st.write("🤖")
with col2:
    st.title("VUCA Smart 🧠")

if 'user_name' not in st.session_state: st.session_state.user_name = ''
if st.session_state.user_name == '':
    st.markdown("### Olá! 👋 Bem-vindo.")
    if n := st.text_input("Como gostaria de ser chamado?"):
        st.session_state.user_name = n
        st.rerun()
    st.stop()

st.markdown(f"Painel de: **{st.session_state.user_name}**")

# --- CARREGAMENTO E ANÁLISE ---
arquivo_vendas_padrao = 'dataset/produtosdevenda-2025-10-13.csv'
arquivo_ficha_padrao = 'dataset/lbox_unidades_cardapio.csv'
vendas = pd.DataFrame()
custos = pd.DataFrame()

if os.path.exists(arquivo_vendas_padrao) and os.path.exists(arquivo_ficha_padrao):
    vendas = filtrar_vendas(arquivo_vendas_padrao)
    custos = processar_nova_ficha(arquivo_ficha_padrao)
else:
    st.warning("Arquivos padrão não encontrados.")
    c1, c2 = st.columns(2)
    up_v = c1.file_uploader("Vendas (CSV)", type=['csv'])
    up_f = c2.file_uploader("Ficha Técnica (CSV)", type=['csv'])
    if up_v: vendas = filtrar_vendas(up_v)
    if up_f: custos = processar_nova_ficha(up_f)

if not vendas.empty and not custos.empty:
    df_final = pd.merge(vendas, custos, on='produto_nome', how='inner')
    
    if not df_final.empty:
        df_final = df_final[df_final['popularidade'] > 0].copy()
        df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
        
        pop_m = df_final['popularidade'].mean()
        luc_m = df_final['lucratividade'].mean()
        df_final['classificacao'] = df_final.apply(lambda x: classificar_produto(x, pop_m, luc_m), axis=1)

        # KPIs
        st.markdown("### 📊 Visão Geral")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total", len(df_final))
        k2.metric("⭐ Estrelas", len(df_final[df_final['classificacao']=='⭐ Estrela']))
        k3.metric("🧩 Quebra-cabeça", len(df_final[df_final['classificacao']=='🧩 Quebra-cabeça']))
        k4.metric("🐶 Cães", len(df_final[df_final['classificacao']=='🐶 Cão']))
        k5.metric("🐴 Burro", len(df_final[df_final['classificacao']=='🐴 Burro de Carga']))

        # Gráfico
        fig = px.scatter(
            df_final, x="popularidade", y="lucratividade", color="classificacao",
            size="popularidade", hover_name="produto_nome",
            color_discrete_map={'⭐ Estrela': '#FFD700', '🐴 Burro de Carga': '#1E90FF', '🧩 Quebra-cabeça': '#32CD32', '🐶 Cão': '#FF4500'},
            template="plotly_white", title="Matriz de Menu"
        )
        fig.add_vline(x=pop_m, line_dash="dash", line_color="gray")
        fig.add_hline(y=luc_m, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🚀 Consultoria IA")
        
        with st.expander("⚙️ Gerar Análise (Clique Aqui)", expanded=False):
            st.info(f"Usando: **{provedor}** ({modelo_selecionado})")
            if st.button("💡 Gerar Recomendações"):
                if not api_key_final:
                    st.error(f"Insira a chave para {provedor}.")
                else:
                    with st.spinner(f"{provedor} analisando..."):
                        try:
                            df_analise = pd.concat([
                                df_final.sort_values('lucratividade', ascending=False).head(15),
                                df_final.sort_values('popularidade', ascending=False).head(15)
                            ]).drop_duplicates().to_csv(index=False, sep=';', decimal=',')
                            
                            res = executar_agente(df_analise, provedor, modelo_selecionado, api_key_final)
                            st.session_state['analise'] = limpar_texto_ia(res)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro: {e}")

        if 'analise' in st.session_state:
            st.success("✅ Relatório Gerado")
            with st.container(border=True):
                st.markdown(st.session_state['analise'])
                if st.button("Limpar"): 
                    del st.session_state['analise']
                    st.rerun()
    else:
        st.warning("Sem dados cruzados.")
else:
    st.info("Aguardando arquivos...")