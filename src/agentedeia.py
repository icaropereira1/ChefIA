from crewai import Agent, Task, Crew, Process, LLM

# --- FUNÇÃO 1: ANÁLISE ESTRATÉGICA DO MENU ---
def executar_analise_menu(df_dados, api_key, modelo_nome):
    
    # Prepara os dados para o prompt
    csv_data = df_dados.to_csv(index=False, sep=';', decimal=',')

    # Configura o LLM
    llm = LLM(model=modelo_nome, api_key=api_key)

    # --- AGENTE 1: O ENGENHEIRO DE MENU (Analítico) ---
    analista = Agent(
        role="Engenheiro de Cardápio Sênior",
        goal="Realizar uma autópsia financeira detalhada, diferenciando produtos de revenda e produção.",
        backstory="""Você é um especialista em CMV (Custo de Mercadoria Vendida) e Engenharia de Menu.
        Você sabe que uma 'Coca-Cola' (Revenda) tem uma lógica financeira totalmente diferente de um 'Risoto' (Produção/Cozinha).
        
        Seu superpoder é identificar quando um item de cozinha está com margem de item de revenda (o que é um erro fatal) 
        ou quando um item de revenda está mal precificado.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # --- AGENTE 2: O CONSULTOR DE LUCRO (Assertivo) ---
    consultor = Agent(
        role="Consultor de Estratégia de Restaurantes",
        goal="Dar ordens diretas e planos de ação claros para o dono do restaurante.",
        backstory="""Você é um consultor pragmático que foca no lucro líquido. 
        Você não usa termos vagos como 'talvez' ou 'considere'. Você diz 'Aumente o preço' ou 'Tire do cardápio'.
        Você traduz a análise técnica em dinheiro no bolso.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # --- TAREFA 1: ANÁLISE PROFUNDA ---
    analisa_performance_cardapio = Task(
        description=f"""
        Analise estes dados de vendas e custos (CSV):
        {csv_data}

        Sua missão é cruzar Popularidade (Vendas) vs Lucratividade (Margem R$) e classificar os itens, 
        MAS com atenção à natureza do produto:

        1. **ITENS DE REVENDA (Ex: Latas, Long Necks, Água, Doces prontos):**
           - Estes itens têm custo fixo e zero mão de obra.
           - Se forem 'Populares' (Burro de Carga), verifique se o preço não está muito baixo. Um aumento de R$ 0,50 aqui gera lucro puro enorme no volume.
           - Se forem 'Críticos' (Cão), sugira combos.

        2. **ITENS DE PRODUÇÃO (Ex: Burgers, Pratos, Sobremesas feitas na casa):**
           - Estes itens exigem gás, mão de obra e tempo.
           - Eles NÃO PODEM ter margem baixa. Se um prato feito é 'Popular' mas tem lucro baixo, isso é uma emergência (Ficha técnica errada ou preço defasado).
           - Identifique 'Oportunidades' (Quebra-cabeça): Pratos de alto lucro que precisam de destaque (foto, descrição).

        Saída esperada: Uma análise técnica que cita NOMES dos produtos e compara seus custos vs preços.
        """,
        expected_output="Relatório técnico detalhando anomalias de precificação e classificação dos itens.",
        agent=analista
    )

    # --- TAREFA 2: PLANO DE AÇÃO ---
    gera_recomendacoes_proativas = Task(
        description="""
        Com base na análise técnica, escreva 3 AÇÕES IMEDIATAS para o dono do restaurante.

        Regras de Ouro:
        1. **Seja Específico:** Não diga "Melhore a margem". Diga "Aumente o preço da Coca-Cola em R$ 1,00".
        2. **Diferencie Estratégia:**
           - Para REVENDAS Populares: Aumente preço (são inelásticos).
           - Para PRODUÇÃO Crítica: Revise a ficha técnica ou remova do cardápio.
           - Para OPORTUNIDADES: Sugira "Sugestão do Chef" ou posts no Instagram.
        3. **Tom de Voz:** Informal, direto e motivador. Use emojis.
        
        Formatação Obrigatória:
        - Use Markdown (negrito, listas).
        - Valores em reais: 'R$ 10,00'.
        - NUNCA use cifrão ($) isolado ou notação LaTeX.
        """,
        expected_output="3 recomendações estratégicas curtas e diretas em Markdown limpo.",
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


# --- FUNÇÃO 2: CHAT RÁPIDO (Mantida simples para velocidade) ---
def responder_chat_dados(pergunta, df_contexto, api_key, modelo_nome):
    
    csv_contexto = df_contexto.to_csv(index=False, sep=';')
    llm = LLM(model=modelo_nome, api_key=api_key)

    agente_chat = Agent(
        role="CFO Virtual",
        goal="Responder perguntas financeiras com precisão baseada nos dados.",
        backstory="Você é um assistente financeiro que tem acesso aos dados do restaurante. Você é direto e numérico.",
        llm=llm,
        verbose=False
    )

    tarefa_chat = Task(
        description=f"""
        Responda à pergunta: '{pergunta}'
        
        Use APENAS estes dados como base:
        {csv_contexto}

        Regras:
        - Se a resposta não estiver nos dados, diga que não sabe.
        - Use formato 'R$ 0,00'.
        - Não use LaTeX ou cifrões ($) soltos.
        """,
        expected_output="Resposta direta em texto simples/markdown.",
        agent=agente_chat
    )

    crew = Crew(agents=[agente_chat], tasks=[tarefa_chat], verbose=False)
    
    return crew.kickoff()