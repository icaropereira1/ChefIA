import pandas as pd
import numpy as np

def processar_nova_ficha(caminho_arquivo):
    """
    Processa o novo arquivo de ficha técnica (lbox_unidades_cardapio.csv)
    para calcular o custo total de produção de cada item.
    """
    print("Iniciando processamento da nova ficha técnica...")
    df_ficha = pd.read_csv(caminho_arquivo, sep=';', encoding='latin1')

    # Renomear colunas para clareza
    df_ficha = df_ficha.rename(columns={
        'produto_principal': 'produto_nome',
        'valor_custo': 'custo_componente'
    })

    # Limpeza de dados numéricos para o custo
    df_ficha['custo_componente'] = pd.to_numeric(df_ficha['custo_componente'], errors='coerce')
    df_ficha = df_ficha.dropna(subset=['custo_componente'])

    # Limpeza robusta do nome do produto (essencial para o merge)
    df_ficha['produto_nome'] = (
        df_ficha['produto_nome']
        .str.strip()
        .str.replace(' +', ' ', regex=True)
        .str.upper()
        .str.rstrip('.') # Remove pontos no final do nome, como em "BISNAGA GARLIC."
    )

    # A lógica principal: agrupar por produto e somar os custos dos componentes
    df_custos = df_ficha.groupby('produto_nome')['custo_componente'].sum().reset_index()

    # Renomear a coluna final de custo
    df_custos = df_custos.rename(columns={'custo_componente': 'custo_producao'})

    print("Nova ficha técnica processada com sucesso!")
    return df_custos


def filtrar_vendas(caminho_do_arquivo):
    """
    Processa o arquivo de vendas, aplicando a mesma limpeza de nomes.
    """
    print("Iniciando processamento dos dados de vendas...")
    df_vendas = pd.read_csv(caminho_do_arquivo, sep=';', encoding='latin1')

    if 'UNIDADE' in df_vendas.columns:
        df_vendas.drop(['UNIDADE'], axis=1, inplace=True)

    df_vendas = df_vendas.rename(columns={
        'PRODUTO DE VENDA': 'produto_nome',
        'VENDA DE FRENTE DE LOJA': 'vendas_loja',
        'VENDA DELIVERY': 'vendas_delivery',
        'RECEITA FRENTE DE LOJA': 'receita_loja',
        'RECEITA DELIVERY': 'receita_delivery'
    })

    # Limpeza robusta do nome do produto (deve ser idêntica à da outra função)
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

    print("Vendas processadas com sucesso!")
    return df_vendas[['produto_nome', 'popularidade', 'preco_venda']]


# ---- Caminhos dos arquivos ----
arquivo_vendas = 'C:/Users/Xícaro PC/pfm_lia/dataset/produtosdevenda-2025-10-13.csv'
arquivo_ficha_novo = 'C:/Users/Xícaro PC/pfm_lia/dataset/lbox_unidades_cardapio.csv'

# ---- Processamento ----
dados_vendas_processados = filtrar_vendas(arquivo_vendas)
dados_custos_processados = processar_nova_ficha(arquivo_ficha_novo)

# ---- Junção dos dados ----
df_final = pd.merge(dados_vendas_processados, dados_custos_processados, on='produto_nome', how='inner')

# ---- Lógica Final ----
if not df_final.empty:
    df_final = df_final[df_final['popularidade'] > 0].copy()
    df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
    df_final = df_final[df_final['lucratividade'] >= 0]

    if not df_final.empty:
        popularidade_media = df_final['popularidade'].mean()
        lucratividade_media = df_final['lucratividade'].mean()

        def classificar_produto(row):
            if row['popularidade'] >= popularidade_media and row['lucratividade'] >= lucratividade_media:
                return '⭐ Estrela'
            elif row['popularidade'] >= popularidade_media and row['lucratividade'] < lucratividade_media:
                return '💪 Burro de Carga'
            elif row['popularidade'] < popularidade_media and row['lucratividade'] >= lucratividade_media:
                return '❓ Quebra-cabeça'
            else:
                return '🐶 Cão'

        df_final['classificacao'] = df_final.apply(classificar_produto, axis=1)

        print("\n--- ANÁLISE FINALIZADA COM SUCESSO! ---")
        print(df_final[['produto_nome', 'popularidade', 'lucratividade', 'classificacao']].head(15))
        print(f"\nTotal de produtos encontrados em comum: {len(df_final)}")
    else:
        print("\n--- ANÁLISE FINALIZADA, MAS NENHUM PRODUTO COM LUCRATIVIDADE POSITIVA FOI ENCONTRADO ---")
else:
    print("\n--- ANÁLISE FINALIZADA, MAS NENHUM PRODUTO EM COMUM FOI ENCONTRADO ---")