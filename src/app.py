import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# --- IMPORTAÇÕES DO SEU NOVO MÓDULO ---
from src.dataloader import processar_nova_ficha, filtrar_vendas, classificar_produto
from src.agentedeia import executar_analise_menu, responder_chat_dados

# --- CARREGA VARIÁVEIS ---
load_dotenv()
st.set_page_config(page_title="VUCA Insights AI", layout="wide")

# --- BARRA LATERAL ---
st.sidebar.title("🔧 Configurações da IA")
provedor = st.sidebar.selectbox("Escolha a Inteligência:", ["Google Gemini", "OpenAI ChatGPT"])

api_key_final = None
modelo_selecionado = None

if provedor == "Google Gemini":
    modelos_google = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    modelo_escolha = st.sidebar.selectbox("Modelo Google:", modelos_google, index=0)
    modelo_selecionado = f"gemini/{modelo_escolha}"
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    api_key_final = env_key if env_key else st.sidebar.text_input("Google API Key:", type="password")

elif provedor == "OpenAI ChatGPT":
    modelos_openai = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    modelo_escolha = st.sidebar.selectbox("Modelo OpenAI:", modelos_openai, index=1)
    modelo_selecionado = f"openai/{modelo_escolha}"
    env_key = os.getenv("OPENAI_API_KEY")
    api_key_final = env_key if env_key else st.sidebar.text_input("OpenAI API Key:", type="password")

if api_key_final:
    if provedor == "Google Gemini":
        os.environ["GOOGLE_API_KEY"] = api_key_final
        os.environ["GEMINI_API_KEY"] = api_key_final
    else:
        os.environ["OPENAI_API_KEY"] = api_key_final
else:
    st.sidebar.warning(f"⚠️ Necessário chave API.")

# --- TELA PRINCIPAL ---
st.title("🤖 VUCA Insights AI")

if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if st.session_state.user_name == '':
    st.markdown("### Olá! 👋 Bem-vindo.")
    name_input = st.text_input("Como gostaria de ser chamado?")
    if name_input:
        st.session_state.user_name = name_input
        st.rerun()
    st.stop()

nome = st.session_state.user_name
st.markdown(f"Painel de Controle de: **{nome}** | 🧠 Modelo: **{modelo_escolha}**")

# --- DADOS ---
arquivo_vendas_padrao = 'produtosdevenda-2025-10-13.csv'
arquivo_ficha_padrao = 'lbox_unidades_cardapio.csv'

if os.path.exists(arquivo_vendas_padrao) and os.path.exists(arquivo_ficha_padrao):
    vendas = filtrar_vendas(arquivo_vendas_padrao)
    custos = processar_nova_ficha(arquivo_ficha_padrao)
else:
    st.warning("Arquivos padrão não encontrados. Faça upload manual.")
    up_vendas = st.sidebar.file_uploader("1. Vendas", type=['csv'])
    up_ficha = st.sidebar.file_uploader("2. Ficha Técnica", type=['csv'])
    vendas = filtrar_vendas(up_vendas) if up_vendas else pd.DataFrame()
    custos = processar_nova_ficha(up_ficha) if up_ficha else pd.DataFrame()

# --- DASHBOARD E IA ---
if not vendas.empty and not custos.empty:
    df_final = pd.merge(vendas, custos, on='produto_nome', how='inner')

    if not df_final.empty:
        df_final = df_final[df_final['popularidade'] > 0].copy()
        df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
        
        pop_media = df_final['popularidade'].mean()
        luc_media = df_final['lucratividade'].mean()

        df_final['classificacao'] = df_final.apply(
            lambda row: classificar_produto(row, pop_media, luc_media), axis=1
        )

        # KPIs e Gráfico
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_final))
        c2.metric("⭐ Estrelas", len(df_final[df_final['classificacao'] == '⭐ Estrela']))
        c3.metric("🧩 Quebra-cabeças", len(df_final[df_final['classificacao'] == '🧩 Quebra-cabeça']))
        c4.metric("🐶 Cães", len(df_final[df_final['classificacao'] == '🐶 Cão']))

        fig = px.scatter(
            df_final, x="popularidade", y="lucratividade", color="classificacao",
            size="popularidade", hover_name="produto_nome",
            color_discrete_map={
                '⭐ Estrela': '#FFD700', '🐴 Burro de Carga': '#1E90FF',
                '🧩 Quebra-cabeça': '#32CD32', '🐶 Cão': '#FF4500'
            }, template="plotly_white"
        )
        fig.add_vline(x=pop_media, line_dash="dash", line_color="gray")
        fig.add_hline(y=luc_media, line_dash="dash", line_color="gray")
        st.plotly_chart(fig)

        st.markdown("---")

        # --- CHAT COM DADOS ---
        st.subheader(f"💬 Chat com seus Dados")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Qual o produto com maior faturamento?"):
            if not api_key_final:
                st.error("⚠️ Configure a API Key primeiro.")
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Prepara dados para o chat
                df_contexto = df_final.sort_values(by='receita_total', ascending=False).head(50)
                
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            # CHAMA A FUNÇÃO DO ARQUIVO src/ai_engine.py
                            resposta = responder_chat_dados(prompt, df_contexto, api_key_final, modelo_selecionado)
                            st.markdown(resposta)
                            st.session_state.messages.append({"role": "assistant", "content": resposta})
                        except Exception as e:
                            st.error(f"Erro: {e}")

        # --- RELATÓRIO COMPLETO ---
        st.markdown("---")
        with st.expander("📝 Gerar Relatório Completo de Consultoria"):
            if st.button("💡 Gerar Recomendações Detalhadas"):
                with st.spinner("Consultando Agente IA..."):
                    try:
                        df_top = df_final.sort_values(by='lucratividade', ascending=False).head(25)
                        
                        # CHAMA A FUNÇÃO DO ARQUIVO src/ai_engine.py
                        res = executar_analise_menu(df_top, api_key_final, modelo_selecionado, nome)
                        
                        st.success("Concluído!")
                        st.markdown(res)
                    except Exception as e:
                        st.error(f"Erro: {e}")
    else:
        st.warning("Sem dados em comum.")