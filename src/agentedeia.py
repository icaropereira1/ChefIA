from crewai import Agent, Task, Crew, Process, LLM

# --- FUNÇÃO 1: ANÁLISE COMPLETA ---
def executar_analise_menu(df_dados, api_key, modelo_nome):
    
    # Prepara os dados para o prompt
    csv_data = df_dados.to_csv(index=False, sep=';', decimal=',')
    llm = LLM(model=modelo_nome, api_key=api_key)

    # --- AGENTE 1: O CÉREBRO TÉCNICO ---
    analista = Agent(
        role="Analista de Engenharia de Cardápio",
        goal="Classificar itens na Matriz de Engenharia de Cardápio e fornecer recomendações financeiras precisas.",
        backstory="""Você é um especialista em Business Intelligence (BI) com 10+ anos de experiência no setor de Restaurantes. 
        Sua especialidade é engenharia de cardápio e otimização de mix de produtos. Você analisa dados friamente e com precisão matemática.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # --- AGENTE 2: O PARCEIRO DE NEGÓCIOS ---
    consultor = Agent(
        role="Consultor de Gestão Operacional",
        goal="Converter análises técnicas em recomendações acionáveis com impacto financeiro claro e legível.",
        backstory="""Você é um consultor de gestão experiente. Você traduz dados brutos em decisões práticas que aumentam lucro. 
        Você escreve de forma limpa e organizada, usando Markdown.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # --- TAREFA 1: ANÁLISE TÉCNICA ---
    analisa_performance_cardapio = Task(
        description=f"""
        Analise estes dados de vendas e custos do restaurante:
        {csv_data}

        Sua tarefa é:
        1. Identificar os produtos 'Estrela', 'Burro de Carga', 'Quebra-cabeça' e 'Cão'.
        2. Calcular métricas de popularidade e lucratividade.
        
        IMPORTANTE SOBRE FORMATAÇÃO:
        - Ao citar valores monetários, use estritamente o formato 'R$ 10,00' (com espaço após o R$).
        - NUNCA use símbolos de dólar duplos ($$) ou simples ($) ao redor de números, pois isso quebra a exibição.
        - Exemplo CORRETO: "Lucro de R$ 500,00"
        - Exemplo ERRADO: "Lucro de $500$"
        """,
        expected_output="Um relatório técnico em Markdown com a classificação dos itens e valores encontrados, sem formatação LaTeX.",
        agent=analista
    )

    # --- TAREFA 2: COMUNICAÇÃO ESTRATÉGICA ---
    gera_recomendacoes_proativas = Task(
        description="""
        Com base no relatório técnico anterior:
        1. Escreva 3 recomendações proativas e urgentes para o dono.
        2. Foque em oportunidades óbvias (ex: aumentar preço do Burro de Carga, promover o Quebra-cabeça).
        
        Escreva em tom informal e direto, como uma mensagem de WhatsApp: 'E aí! Notei algumas coisas aqui...'
        
        REGRAS CRÍTICAS DE FORMATAÇÃO:
        - Use Markdown padrão para negrito (**texto**) e listas (- item).
        - Ao escrever valores em reais, use sempre 'R$ ' (com espaço).
        - PROIBIDO usar cifrões para indicar fórmulas matemáticas. Escreva o valor como texto normal.
        """,
        expected_output="Texto curto e informal com 3 recomendações práticas em Markdown limpo e legível.",
        agent=consultor,
        context=[analisa_performance_cardapio]
    )

    crew = Crew(
        agents=[analista, consultor],
        tasks=[analisa_performance_cardapio, gera_recomendacoes_proativas],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


# --- FUNÇÃO 2: CHAT RÁPIDO ---
def responder_chat_dados(pergunta, df_contexto, api_key, modelo_nome):
    
    csv_contexto = df_contexto.to_csv(index=False, sep=';')
    llm = LLM(model=modelo_nome, api_key=api_key)

    agente_chat = Agent(
        role="Assistente de Dados Financeiros",
        goal="Responder perguntas pontuais sobre o faturamento de forma clara e sem erros de formatação.",
        backstory="Você é um assistente prestativo com acesso aos dados de vendas. Você prioriza a clareza na comunicação.",
        llm=llm,
        verbose=False
    )

    tarefa_chat = Task(
        description=f"""
        Responda à pergunta do usuário com base APENAS nestes dados:
        {csv_contexto}

        PERGUNTA: {pergunta}
        
        DIRETRIZES DE RESPOSTA:
        - Seja direto e use Markdown para destacar pontos importantes.
        - Ao citar dinheiro, use o formato 'R$ 10,00'.
        - EVITE completamente o uso do símbolo '$' para qualquer coisa que não seja o prefixo da moeda 'R$'. 
        - O sistema de exibição não suporta LaTeX, então escreva tudo como texto simples.
        """,
        expected_output="Resposta direta e conversacional em Markdown limpo, sem caracteres especiais de matemática.",
        agent=agente_chat
    )

    crew = Crew(agents=[agente_chat], tasks=[tarefa_chat], verbose=False)
    
    return crew.kickoff()