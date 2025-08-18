# utils/analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_correlation_heatmap(csv_file_path, output_png_path):
    """
    Carrega dados de um CSV, gera um heatmap de correlação e o salva em um arquivo PNG.
    Esta função foi projetada para ser chamada de forma automatizada.
    
    Args:
        csv_file_path (str): O caminho para o arquivo CSV com os dados da otimização.
        output_png_path (str): O caminho onde o arquivo PNG do heatmap será salvo.
    """
    try:
        # Verifica se o arquivo de dados existe
        if not os.path.exists(csv_file_path):
            #print(f"  [Análise] Arquivo de dados '{os.path.basename(csv_file_path)}' ainda não criado. Pulando heatmap.")
            return

        df = pd.read_csv(csv_file_path)
        
        # Correlação requer pelo menos 2 amostras com alguma variação
        if len(df) < 2 or df['delta_amp'].nunique() < 2:
            #print(f"  [Análise] Dados insuficientes para gerar correlação (amostras={len(df)}). Pulando heatmap.")
            return

        # Filtra resultados inválidos que podem ter vindo de simulações falhas
        df = df[df['delta_amp'] > -1e30]
        
        params_and_fitness = ['s', 'w', 'l', 'height', 'delta_amp']
        df_corr = df[params_and_fitness]
        
        correlation_matrix = df_corr.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5
        )
        plt.title(f'Heatmap de Correlação (Atualizado na Geração {df["generation"].max()})')
        
        # Salva a figura no caminho especificado
        plt.savefig(output_png_path)
        
        # Fecha a figura para liberar memória e evitar que ela seja exibida na tela
        plt.close()

    except Exception as e:
        print(f"!!! Erro ao gerar o heatmap de correlação: {e}")

# Você pode manter esta parte se quiser rodar a análise completa manualmente
if __name__ == '__main__':
    print("Este script foi projetado para ser importado como um módulo.")
    print("Para análise manual, execute uma versão anterior ou crie uma função de chamada aqui.")