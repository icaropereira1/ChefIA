import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os # ### NOVO: Para gerenciar a chave de API

# ### NOVO: Importar as bibliotecas do Agente ###
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI # Para usar o Gemini

# --- INÍCIO DA LÓGICA DE ANÁLISE ---
# ### NOVO: Adicionamos @st.cache_data para acelerar o processamento ###
# Isso evita que o app re-processe os mesmos arquivos toda vez que algo muda na tela
@st.cache_data
def processar_nova_ficha(arquivo_ficha):
    df_ficha = pd.read_csv(arquivo_ficha, sep=';', encoding='latin1')
    df_ficha = df_ficha.rename(columns={
        'produto_principal': 'produto_nome',
        'valor_custo': 'custo_componente'
    })
    df_ficha['custo_componente'] = pd.to_numeric(df_ficha['custo_componente'], errors='coerce')
    df_ficha = df_ficha.dropna(subset=['custo_componente'])
    df_ficha['produto_nome'] = (
        df_ficha['produto_nome']
        .str.strip()
        .str.replace(' +', ' ', regex=True)
        .str.upper()
        .str.rstrip('.')
    )
    df_custos = df_ficha.groupby('produto_nome')['custo_componente'].sum().reset_index()
    df_custos = df_custos.rename(columns={'custo_componente': 'custo_producao'})
    return df_custos

@st.cache_data
def filtrar_vendas(arquivo_vendas):
    df_vendas = pd.read_csv(arquivo_vendas, sep=';', encoding='latin1')
    if 'UNIDADE' in df_vendas.columns:
        df_vendas.drop(['UNIDADE'], axis=1, inplace=True)
    df_vendas = df_vendas.rename(columns={
        'PRODUTO DE VENDA': 'produto_nome',
        'VENDA DE FRENTE DE LOJA': 'vendas_loja',
        'VENDA DELIVERY': 'vendas_delivery',
        'RECEITA FRENTE DE LOJA': 'receita_loja',
        'RECEITA DELIVERY': 'receita_delivery'
    })
    df_vendas['produto_nome'] = (
        df_vendas['produto_nome']
        .str.strip()
        .str.replace(' +', ' ', regex=True)
        .str.upper()
    )
    colunas_numericas = ['vendas_loja', 'vendas_delivery', 'receita_loja', 'receita_delivery']
    for col in colunas_numericas:
        df_vendas[col] = pd.to_numeric(
            df_vendas[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
            errors='coerce'
        )
    df_vendas = df_vendas.fillna(0)
    df_vendas['popularidade'] = df_vendas['vendas_loja'] + df_vendas['vendas_delivery']
    df_vendas['receita_total'] = df_vendas['receita_loja'] + df_vendas['receita_delivery']
    df_vendas['preco_venda'] = np.where(df_vendas['popularidade'] > 0,
                                          df_vendas['receita_total'] / df_vendas['popularidade'], 0)
    return df_vendas[['produto_nome', 'popularidade', 'preco_venda']]

def classificar_produto(row, popularidade_media, lucratividade_media):
    if row['popularidade'] >= popularidade_media and row['lucratividade'] >= lucratividade_media:
        return '⭐ Estrela'
    elif row['popularidade'] >= popularidade_media and row['lucratividade'] < lucratividade_media:
        return '💪 Burro de Carga'
    elif row['popularidade'] < popularidade_media and row['lucratividade'] >= lucratividade_media:
        return '❓ Quebra-cabeça'
    else:
        return '🐶 Cão'

# --- FIM DA LÓGICA DE ANÁLISE ---


# --- INTERFACE DO STREAMLIT ---

st.set_page_config(layout="wide")
st.title('🤖 VUCA Insights AI - Analisador e Agente Proativo')

# ### NOVO: Seção para inserir a chave de API do Google ###
st.sidebar.header('Configuração do Agente AI')
api_key = st.sidebar.text_input("Sua Chave de API do Google AI (Gemini)", type="password")

llm = None
if api_key:
    # Configura a chave de API como variável de ambiente para o crewai usar
    os.environ["GOOGLE_API_KEY"] = api_key 
    try:
        # Inicializa o modelo de linguagem (LLM) que os agentes usarão
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
        st.sidebar.success("API Key configurada com sucesso!")
    except Exception as e:
        st.sidebar.error(f"Erro ao configurar a API: {e}")
else:
    st.sidebar.warning("Insira sua chave de API do Google para habilitar os Agentes de IA.")


# --- UPLOAD DOS ARQUIVOS (Como antes) ---
st.sidebar.header('Carregue seus arquivos aqui:')
uploaded_vendas = st.sidebar.file_uploader("1. Arquivo de Vendas (Ex: produtosdevenda...)", type=['csv'])
uploaded_ficha = st.sidebar.file_uploader("2. Arquivo de Ficha Técnica (Ex: lbox...)", type=['csv'])

if uploaded_vendas is not None and uploaded_ficha is not None:
    try:
        # --- PROCESSAMENTO E ANÁLISE (Como antes) ---
        dados_vendas = filtrar_vendas(uploaded_vendas)
        dados_custos = processar_nova_ficha(uploaded_ficha)
        df_final = pd.merge(dados_vendas, dados_custos, on='produto_nome', how='inner')

        if not df_final.empty:
            df_final = df_final[df_final['popularidade'] > 0].copy()
            df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
            df_final = df_final[df_final['lucratividade'] >= 0]

            if not df_final.empty:
                pop_media = df_final['popularidade'].mean()
                luc_media = df_final['lucratividade'].mean()
                df_final['classificacao'] = df_final.apply(
                    lambda row: classificar_produto(row, pop_media, luc_media), axis=1
                )

                st.success(f"Análise concluída! {len(df_final)} produtos em comum foram encontrados e analisados.")

                # --- VISUALIZAÇÃO DOS RESULTADOS (Como antes) ---
                st.header("Dashboard de Análise de Cardápio")
                fig = px.scatter(
                    df_final,
                    x="popularidade", y="lucratividade", color="classificacao",
                    size="popularidade", hover_name="produto_nome",
                    title="Matriz de Performance dos Produtos",
                    labels={"popularidade": "Popularidade (Nº de Vendas)", "lucratividade": "Lucratividade (R$ por Venda)"},
                    color_discrete_map={
                        '⭐ Estrela': 'gold', '💪 Burro de Carga': 'blue',
                        '❓ Quebra-cabeça': 'green', '🐶 Cão': 'red'
                    }
                )
                fig.add_vline(x=pop_media, line_dash="dash", line_color="gray", annotation_text="Média de Popularidade")
                fig.add_hline(y=luc_media, line_dash="dash", line_color="gray", annotation_text="Média de Lucratividade")
                st.plotly_chart(fig, use_container_width=True)

                # --- Detalhes e Recomendações (Como antes) ---
                st.header("Recomendações Estratégicas")
                col1, col2 = st.columns(2)
                # ... (Seu código das tabelas de Estrelas, Cães, etc. vai aqui)
                with col1:
                    st.subheader("⭐ Estrelas")
                    st.markdown("Alta Lucratividade e Alta Popularidade. **Ação:** Destaque-os e mantenha a qualidade!")
                    st.dataframe(df_final[df_final['classificacao'] == '⭐ Estrela'])

                    st.subheader("❓ Quebra-cabeças")
                    st.markdown("Alta Lucratividade e Baixa Popularidade. **Ação:** Promova! Treine a equipe para sugeri-los.")
                    st.dataframe(df_final[df_final['classificacao'] == '❓ Quebra-cabeça'])

                with col2:
                    st.subheader("💪 Burros de Carga")
                    st.markdown("Baixa Lucratividade e Alta Popularidade. **Ação:** Essenciais, mas tente otimizar os custos ou aumentar o preço.")
                    st.dataframe(df_final[df_final['classificacao'] == '💪 Burro de Carga'])

                    st.subheader("🐶 Cães")
                    st.markdown("Baixa Lucratividade e Baixa Popularidade. **Ação:** Analise a possibilidade de remover, simplificar ou repaginar.")
                    st.dataframe(df_final[df_final['classificacao'] == '🐶 Cão'])
                

                # ### NOVO: Seção do Agente Proativo ###
                st.markdown("---")
                st.header("🤖 Agente Proativo de Insights")
                
                if not llm:
                    st.warning("Insira sua chave de API na barra lateral para usar o Agente.")
                else:
                    if st.button("Executar Agente AI para Gerar Recomendações"):
                        
                        # 1. Inserindo os dados: Convertemos o DataFrame em uma string CSV
                        dados_em_string = df_final.to_csv(index=False, sep=';', decimal=',')
                        
                        # 2. Definição dos Agentes (do seu notebook)
                        analista_de_dados = Agent(
                            role="Analista de Engenharia de Cardápio",
                            goal="Analisar dados de vendas e custos para classificar itens de cardápio.",
                            backstory="Especialista em BI para restaurantes, focado em identificar performance de produtos (Estrela, Burro de Carga, Quebra-cabeça, Cão).",
                            verbose=True,
                            llm=llm, # Usa o LLM que inicializamos
                            allow_delegation=False
                        )
                        estrategista_de_gestao = Agent(
                            role="Consultor e Estrategista de Restaurante",
                            goal="Transformar análises de dados em recomendações de negócio acionáveis.",
                            backstory="Ex-dono de restaurante que traduz dados complexos em ações simples e diretas (preço, promoção, marketing) para aumentar o lucro.",
                            verbose=True,
                            llm=llm, # Usa o LLM que inicializamos
                            allow_delegation=False
                        )
                        
                        # 3. Definição das Tarefas (do seu notebook)
                        analisa_performance_cardapio = Task(
                            description=f"""Analise estes dados de cardápio de um restaurante.
                            Os dados estão em formato CSV com ';' como separador.
                            Sua tarefa é identificar os 2 principais 'Quebra-cabeças' (alta lucratividade, baixa popularidade)
                            e os 2 principais 'Burros de Carga' (baixa lucratividade, alta popularidade).
                            Liste-os com seus nomes, popularidade e lucratividade.
                            
                            DADOS:
                            {dados_em_string}
                            """,
                            expected_output="Um relatório técnico listando os 2 principais 'Quebra-cabeças' e os 2 principais 'Burros de Carga' com seus valores.",
                            agent=analista_de_dados
                        )
                        
                        gera_recomendacoes_proativas = Task(
                            description="""Com base no relatório de análise de 'Quebra-cabeças' e 'Burros de Carga',
                            escreva 3 recomendações proativas para o dono do restaurante.
                            Escreva em tom informal e direto (como no WhatsApp).
                            Seja específico sobre os pratos.
                            Comece com 'E aí! Sou seu assistente VUCA AI e notei algumas coisas importantes:'""",
                            expected_output="Um texto em português claro, com 3 recomendações de negócio.",
                            agent=estrategista_de_gestao,
                            context=[analisa_performance_cardapio] # Depende da primeira tarefa
                        )
                        
                        # 4. Execução da Crew
                        insights_restaurante_crew = Crew(
                            agents=[analista_de_dados, estrategista_de_gestao],
                            tasks=[analisa_performance_cardapio, gera_recomendacoes_proativas],
                            process=Process.sequential
                        )
                        
                        with st.spinner("Os agentes de IA estão analisando e gerando as recomendações..."):
                            # Usamos kickoff SEM 'inputs' porque já passamos os dados direto na descrição da Tarefa 1
                            resultado = insights_restaurante_crew.kickoff()
                        
                        st.subheader("Recomendações Proativas do Agente:")
                        st.markdown(resultado)

            else:
                st.warning("A análise foi concluída, mas nenhum produto com lucratividade positiva foi encontrado.")
        else:
            st.error("Nenhum produto em comum foi encontrado. Verifique os arquivos.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a análise: {e}")
else:
    st.info("Por favor, carregue os dois arquivos CSV na barra lateral para iniciar a análise.")