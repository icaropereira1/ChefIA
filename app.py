import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# --- IMPORTAÇÕES DOS MÓDULOS ---
# Certifique-se de que os arquivos na pasta 'src' têm exatamente estes nomes:
# src/data_loader.py e src/ai_engine.py
from src.dataloader import processar_nova_ficha, filtrar_vendas, classificar_produto
from src.agentedeia import executar_analise_menu, responder_chat_dados

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
st.set_page_config(page_title="Vuca Smart", layout="wide")

# --- FUNÇÃO DE LIMPEZA DE TEXTO (CORREÇÃO DO BUG VISUAL) ---
def limpar_texto_ia(texto_obj):
    """
    Converte o resultado da IA para string e escapa o cifrão ($).
    Isso impede que o Streamlit tente criar fórmulas matemáticas (LaTeX) onde não deve.
    """
    # 1. Extrai o texto se for um objeto CrewOutput
    if hasattr(texto_obj, 'raw'):
        texto = str(texto_obj.raw)
    else:
        texto = str(texto_obj)
    
    # 2. Substitui $ por \$ (Cifrão literal)
    # Isso corrige o bug da fonte estranha
    return texto.replace("$", "\\$")

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
    st.sidebar.warning(f"⚠️ Necessário chave API para ativar a IA.")


# --- TELA PRINCIPAL ---
st.title("VUCA Smart 🧠")

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
st.markdown(f"Painel de Controle de: **{nome}**")


# --- CARREGAMENTO DE DADOS ---
arquivo_vendas_padrao = 'dataset/produtosdevenda-2025-10-13.csv'
arquivo_ficha_padrao = 'dataset/lbox_unidades_cardapio.csv'

if os.path.exists(arquivo_vendas_padrao) and os.path.exists(arquivo_ficha_padrao):
    vendas = filtrar_vendas(arquivo_vendas_padrao)
    custos = processar_nova_ficha(arquivo_ficha_padrao)
else:
    st.warning("Arquivos padrão não encontrados na pasta 'dataset'. Faça upload manual.")
    c1, c2 = st.columns(2)
    up_vendas = c1.file_uploader("1. Vendas (CSV)", type=['csv'])
    up_ficha = c2.file_uploader("2. Ficha Técnica (CSV)", type=['csv'])
    
    vendas = filtrar_vendas(up_vendas) if up_vendas else pd.DataFrame()
    custos = processar_nova_ficha(up_ficha) if up_ficha else pd.DataFrame()


# --- DASHBOARD ---
if not vendas.empty and not custos.empty:
    df_final = pd.merge(vendas, custos, on='produto_nome', how='inner')

    if not df_final.empty:
        # Cálculos
        df_final = df_final[df_final['popularidade'] > 0].copy()
        df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
        
        pop_media = df_final['popularidade'].mean()
        luc_media = df_final['lucratividade'].mean()

        df_final['classificacao'] = df_final.apply(
            lambda row: classificar_produto(row, pop_media, luc_media), axis=1
        )

        # 1. SEÇÃO DE KPIs
        st.markdown("### 📊 Visão Geral do Cardápio")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_final))
        c2.metric("⭐ Estrelas", len(df_final[df_final['classificacao'] == '⭐ Estrela']))
        c3.metric("🧩 Quebra-cabeças", len(df_final[df_final['classificacao'] == '🧩 Quebra-cabeça']))
        c4.metric("🐶 Cães", len(df_final[df_final['classificacao'] == '🐶 Cão']))

        # 2. SEÇÃO DE GRÁFICOS
        fig = px.scatter(
            df_final, x="popularidade", y="lucratividade", color="classificacao",
            size="popularidade", hover_name="produto_nome",
            color_discrete_map={
                '⭐ Estrela': '#FFD700', '🐴 Burro de Carga': '#1E90FF',
                '🧩 Quebra-cabeça': '#32CD32', '🐶 Cão': '#FF4500'
            }, 
            template="plotly_white",
            title="Matriz de Engenharia de Menu (Popularidade x Lucratividade)"
        )
        fig.add_vline(x=pop_media, line_dash="dash", line_color="gray", annotation_text="Média Popularidade")
        fig.add_hline(y=luc_media, line_dash="dash", line_color="gray", annotation_text="Média Lucro")
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # 3. SEÇÃO DE CONSULTORIA ESTRATÉGICA (MOVIDO PARA CIMA)
        st.subheader("🚀 Plano de Ação Estratégico")
        
        with st.expander("⚙️ Gerar Nova Consultoria (Clique Aqui)", expanded=False):
            st.info("Esta análise usa agentes avançados para calcular cenários. Pode levar alguns segundos.")
            
            if st.button("💡 Gerar Recomendações Detalhadas"):
                if not api_key_final:
                    st.error("⚠️ Configure a API Key primeiro.")
                else:
                    with st.spinner(f"Os Agentes de IA estão trabalhando ({provedor})..."):
                        try:
                            df_top_lucro = df_final.sort_values(by='lucratividade', ascending=False).head(15)
                            df_top_venda = df_final.sort_values(by='popularidade', ascending=False).head(15)
                            df_analise = pd.concat([df_top_lucro, df_top_venda]).drop_duplicates()
                            
                            res = executar_analise_menu(df_analise, api_key_final, modelo_selecionado)
                            
                            # Salva usando a função de limpeza para corrigir o texto
                            st.session_state['analise_completa'] = limpar_texto_ia(res)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Erro na execução da IA: {e}")

        # Exibição do Relatório (Persistente e Corrigida)
        if 'analise_completa' in st.session_state:
            st.success("✅ Relatório Gerado")
            with st.container(border=True):
                # Aqui o texto já vem limpo do session_state, mas garantimos
                st.markdown(st.session_state['analise_completa'])
                
                st.markdown("---")
                if st.button("🗑️ Limpar Relatório"):
                    del st.session_state['analise_completa']
                    st.rerun()

        st.markdown("---")

        # 4. SEÇÃO DE CHAT (MOVIDO PARA BAIXO)
        st.subheader(f"💬 Chat com seus Dados")
        st.caption("Pergunte sobre faturamento, margens ou detalhes dos produtos.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Corrige formatação ao exibir histórico
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Qual o produto com maior margem de lucro?"):
            if not api_key_final:
                st.error("⚠️ Configure a API Key na barra lateral primeiro.")
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                df_contexto = df_final.sort_values(by='receita_total', ascending=False).head(50)
                
                with st.chat_message("assistant", avatar="dataset/logovuca.png"):
                    with st.spinner("Analisando dados..."):
                        try:
                            resposta_obj = responder_chat_dados(prompt, df_contexto, api_key_final, modelo_selecionado)
                            
                            # Limpa o texto antes de salvar ou exibir
                            resposta_limpa = limpar_texto_ia(resposta_obj)
                            
                            st.markdown(resposta_limpa)
                            st.session_state.messages.append({"role": "assistant", "content": resposta_limpa})
                        except Exception as e:
                            st.error(f"Erro ao processar: {e}")

    else:
        st.warning("Atenção: O cruzamento dos arquivos não gerou dados. Verifique se os nomes dos produtos são iguais nos dois arquivos.")

else:
    if not vendas.empty:
        st.info("✅ Arquivo de Vendas carregado. Aguardando Ficha Técnica...")
    elif not custos.empty:
        st.info("✅ Ficha Técnica carregada. Aguardando Arquivo de Vendas...")
    else:
        st.info("👋 Aguardando carregamento dos arquivos de dados.")