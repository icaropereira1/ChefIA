import pandas as pd
import numpy as np

def processar_nova_ficha(arquivo):
    try:
        df_ficha = pd.read_csv(arquivo, sep=';', encoding='latin1')
        df_ficha.columns = df_ficha.columns.str.replace('"', '').str.strip().str.lower()

        if 'valor_custo' not in df_ficha.columns and 'valor custo' not in df_ficha.columns:
            return pd.DataFrame()

        mapa_colunas = {
            'produto_principal': 'produto_nome', 'produto principal': 'produto_nome',
            'valor_custo': 'custo_componente', 'valor custo': 'custo_componente'
        }
        df_ficha = df_ficha.rename(columns=mapa_colunas)
        df_ficha['custo_componente'] = pd.to_numeric(df_ficha['custo_componente'], errors='coerce')
        df_ficha = df_ficha.dropna(subset=['custo_componente'])

        if 'produto_nome' in df_ficha.columns:
            df_ficha['produto_nome'] = (
                df_ficha['produto_nome'].astype(str).str.strip().str.replace(' +', ' ', regex=True).str.upper().str.rstrip('.')
            )
            df_custos = df_ficha.groupby('produto_nome')['custo_componente'].sum().reset_index()
            df_custos = df_custos.rename(columns={'custo_componente': 'custo_producao'})
            return df_custos

        return pd.DataFrame()
    except:
        return pd.DataFrame()

def filtrar_vendas(arquivo):
    try:
        df_vendas = pd.read_csv(arquivo, sep=';', encoding='latin1')
        df_vendas.columns = df_vendas.columns.str.replace('"', '').str.strip().str.upper()

        if 'PRODUTO DE VENDA' not in df_vendas.columns:
            return pd.DataFrame()

        if 'UNIDADE' in df_vendas.columns:
            df_vendas.drop(['UNIDADE'], axis=1, inplace=True)

        df_vendas = df_vendas.rename(columns={
            'PRODUTO DE VENDA': 'produto_nome', 'VENDA DE FRENTE DE LOJA': 'vendas_loja',
            'VENDA DELIVERY': 'vendas_delivery', 'RECEITA FRENTE DE LOJA': 'receita_loja',
            'RECEITA DELIVERY': 'receita_delivery'
        })

        df_vendas['produto_nome'] = (
            df_vendas['produto_nome'].astype(str).str.strip().str.replace(' +', ' ', regex=True).str.upper()
        )

        for col in ['vendas_loja', 'vendas_delivery', 'receita_loja', 'receita_delivery']:
            if col in df_vendas.columns:
                df_vendas[col] = pd.to_numeric(
                    df_vendas[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                    errors='coerce'
                )

        df_vendas = df_vendas.fillna(0)
        df_vendas['popularidade'] = df_vendas['vendas_loja'] + df_vendas['vendas_delivery']
        df_vendas['receita_total'] = df_vendas['receita_loja'] + df_vendas['receita_delivery']
        
        df_vendas['preco_venda'] = np.where(
            df_vendas['popularidade'] > 0,
            df_vendas['receita_total'] / df_vendas['popularidade'],
            0
        )

        return df_vendas[['produto_nome', 'popularidade', 'preco_venda', 'receita_total']]
    except:
        return pd.DataFrame()

def classificar_produto(row, pop_media, luc_media):
    if row['popularidade'] >= pop_media and row['lucratividade'] >= luc_media:
        return '⭐ Estrela'
    elif row['popularidade'] >= pop_media and row['lucratividade'] < luc_media:
        return '🐴 Burro de Carga'
    elif row['popularidade'] < pop_media and row['lucratividade'] >= luc_media:
        return '🧩 Quebra-cabeça'
    else:
        return '🐶 Cão'