from datetime import datetime
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Definir caminhos dos dados
ENDERECO_DADOS = r'../bronze/'
ENDERECO_VOTACAO = r'../votacao/'

def carregar_dados():
    """Carrega dados do Bolsa Família e da votação do 2º turno."""
    try:
        print('Carregando dados...')
        inicio = datetime.now()

        df_bolsa = pl.read_parquet(ENDERECO_DADOS + 'bolsa_familia_str_cache.parquet')
        df_votos = pl.read_csv(ENDERECO_VOTACAO + 'votacao_secao_2022_BR.csv', separator=';', encoding='iso-8859-1')

        print(f'Dados carregados em {datetime.now() - inicio}')
        return df_bolsa, df_votos
    except Exception as e:
        raise RuntimeError(f'Erro ao carregar os dados: {e}')

def preparar_dados(df_bolsa, df_votos):
    """Filtra e agrupa os dados para análise."""
    try:
        print('Preparando dados...')
        inicio = datetime.now()

        # Filtrar votação 2º turno e candidatos Lula (13) e Bolsonaro (22)
        df_votos_filtrado = df_votos.filter(
            (pl.col('NR_TURNO') == 2) &
            (pl.col('NR_VOTAVEL').is_in([13, 22]))
        )

        with pl.StringCache():
            # Selecionar e categorizar colunas importantes
            df_votos_sel = df_votos_filtrado.select(['SG_UF', 'NM_VOTAVEL', 'QT_VOTOS'])
            df_votos_sel = df_votos_sel.with_columns([
                pl.col('SG_UF').cast(pl.Categorical),
                pl.col('NM_VOTAVEL').cast(pl.Categorical)
            ])

            # Agrupar votos por estado e candidato
            df_votos_agg = df_votos_sel.group_by(['SG_UF', 'NM_VOTAVEL']).agg(
                pl.col('QT_VOTOS').sum().alias('QT_VOTOS')
            )

            # Preparar dados do Bolsa Família
            df_bolsa_sel = df_bolsa.select(['UF', 'VALOR PARCELA'])
            df_bolsa_sel = df_bolsa_sel.with_columns([
                pl.col('UF').cast(pl.Categorical)
            ])
            df_bolsa_agg = df_bolsa_sel.group_by('UF').agg(
                pl.col('VALOR PARCELA').sum().alias('VALOR_PARCECLA_TOTAL')
            )

            # Juntar votação com Bolsa Família por estado
            df_completo = df_votos_agg.join(df_bolsa_agg, left_on='SG_UF', right_on='UF')

        print(f'Dados preparados em {datetime.now() - inicio}')
        return df_completo

    except Exception as e:
        raise RuntimeError(f'Erro na preparação dos dados: {e}')

def totalizar_votos(df_votos):
    """Calcula total de votos por candidato e resumo nacional."""
    print('Totalizando votos...')
    inicio = datetime.now()

    total_por_candidato = df_votos.group_by('NM_VOTAVEL').agg(
        pl.col('QT_VOTOS').sum().alias('TOTAL_VOTOS')
    ).sort('TOTAL_VOTOS', descending=True)

    total_geral = total_por_candidato['TOTAL_VOTOS'].sum()
    total_por_candidato = total_por_candidato.with_columns([
        (pl.col('TOTAL_VOTOS') / total_geral * 100).alias('PERCENTUAL')
    ])

    print(f'Totalização concluída em {datetime.now() - inicio}')
    return total_por_candidato, total_geral

def calcular_correlacao(df_completo):
    """Calcula correlação votos x Bolsa Família para cada candidato."""
    print('Calculando correlações...')
    inicio = datetime.now()

    correlacoes = {}

    for candidato in df_completo['NM_VOTAVEL'].unique():
        df_cand = df_completo.filter(pl.col('NM_VOTAVEL') == candidato)
        votos = np.array(df_cand['QT_VOTOS'])
        bolsa = np.array(df_cand['VALOR_PARCECLA_TOTAL'])
        corr = np.corrcoef(votos, bolsa)[0, 1]
        correlacoes[candidato] = corr
        print(f'Correlação {candidato}: {corr:.3f}')

    print(f'Correlações calculadas em {datetime.now() - inicio}')
    return correlacoes

def plotar_graficos(df_completo, correlacoes, total_por_candidato, total_geral):
    """Gera gráficos da análise."""
    print('Gerando gráficos...')
    inicio = datetime.now()

    candidatos = list(correlacoes.keys())

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análise da Votação do 2º Turno e Bolsa Família por Estado', fontsize=16)

    # Gráfico 1 e 2: Votos por estado para cada candidato
    for i, candidato in enumerate(candidatos):
        ax = axs[0, i]
        df_cand = df_completo.filter(pl.col('NM_VOTAVEL') == candidato).sort('QT_VOTOS', descending=True)
        ax.bar(df_cand['SG_UF'], df_cand['QT_VOTOS'], color='skyblue')
        ax.set_title(f'Votos por Estado - {candidato}')
        ax.set_xlabel('Estado (UF)')
        ax.set_ylabel('Número de Votos')
        ax.tick_params(axis='x', rotation=45)

    # Gráfico 3: Valores pagos Bolsa Família por estado
    ax3 = axs[1, 0]
    df_bolsa = df_completo.select(['SG_UF', 'VALOR_PARCECLA_TOTAL']).unique().sort('VALOR_PARCECLA_TOTAL', descending=True)
    ax3.bar(df_bolsa['SG_UF'], df_bolsa['VALOR_PARCECLA_TOTAL'], color='orange')
    ax3.set_title('Valores pagos do Bolsa Família por Estado')
    ax3.set_xlabel('Estado (UF)')
    ax3.set_ylabel('Valor Total Pago')
    ax3.tick_params(axis='x', rotation=45)

    # Gráfico 4: Texto com correlações e resumo de votos
    ax4 = axs[1, 1]
    ax4.axis('off')
    texto = f'Resumo Nacional 2º Turno:\n\n'
    for idx, row in enumerate(total_por_candidato.rows()):
        nome, votos, perc = row
        texto += f'{nome}: {int(votos):,} votos ({perc:.2f}%)\n'
    texto += f'\nTotal de votos válidos: {int(total_geral):,}\n\nCorrelação votos x Bolsa Família:\n'
    for cand, corr in correlacoes.items():
        texto += f'{cand}: {corr:.3f}\n'
    ax4.text(0, 1, texto, fontsize=12, verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(f'Gráficos gerados em {datetime.now() - inicio}')

def main():
    """Função principal para executar todas as etapas."""
    df_bolsa, df_votos = carregar_dados()
    df_completo = preparar_dados(df_bolsa, df_votos)
    total_por_candidato, total_geral = totalizar_votos(df_completo)
    correlacoes = calcular_correlacao(df_completo)
    plotar_graficos(df_completo, correlacoes, total_por_candidato, total_geral)

if __name__ == '__main__':
    main()