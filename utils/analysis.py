# utils/analysis.py (Modificado)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_full_analysis(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        print("Dados carregados com sucesso!")
        print(f"Total de indivíduos analisados: {len(df)}")
        
        df = df[df['fitness_score'] > -1e30]
        
        # [MODIFICADO] Adicionada a coluna 'total_length' para análise
        params_and_fitness = ['s', 'w', 'l', 'height', 'total_length', 'fitness_score']
        df_analysis = df[params_and_fitness]

        output_directory = os.path.dirname(csv_file_path)
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        heatmap_output_path = os.path.join(output_directory, f"{base_filename}_heatmap.png")
        pairplot_output_path = os.path.join(output_directory, f"{base_filename}_pairplot.png")

        print(f"\nGerando Heatmap de Correlação...")
        correlation_matrix = df_analysis.corr()
        
        plt.figure(figsize=(12, 10)) # Aumentei um pouco o tamanho para a nova variável
        sns.heatmap(
            correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5
        )
        plt.title('Matriz de Correlação entre Parâmetros e Fitness (fitness_score)')
        
        plt.savefig(heatmap_output_path)
        plt.close()
        print(f"-> Heatmap salvo em: {heatmap_output_path}")
        
        print("\nGerando Pairplot... Isso pode ser demorado com mais variáveis.")
        
        pair_plot = sns.pairplot(
            df_analysis,
            diag_kind='kde' 
        )
        
        pair_plot.fig.suptitle('Análise Visual de Pares entre Parâmetros e Fitness', y=1.02)
        
        pair_plot.savefig(pairplot_output_path)
        plt.close()
        print(f"-> Pairplot salvo em: {pairplot_output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file_path}' não foi encontrado.")
    except KeyError as e:
        print(f"Erro de Chave durante a análise: a coluna {e} não foi encontrada no CSV.")
        print("Verifique se o nome da coluna ('fitness_score', 'total_length', etc.) está correto em analysis.py.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")