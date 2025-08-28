# S_matrix_calculations.py (Modificado para retornar S11 médio)

import numpy as np

# Definimos o comprimento total do guia de onda (constante)
GUIDE_LENGTH = 40e-6

def _s_to_t_matrix_conversion(S_matrix):
    """
    Converte uma matriz S de 2x2 para uma matriz T de 2x2.
    USA FÓRMULAS PADRÃO PARA REDES RECÍPROCAS.
    """
    S11 = S_matrix[0, 0]
    S12 = S_matrix[0, 1]
    S21 = S_matrix[1, 0]
    S22 = S_matrix[1, 1]

    if np.abs(S21) < 1e-12:
        return np.identity(2, dtype=np.complex128)

    det_S = S11 * S22 - S12 * S21
    T11 = -det_S / S21
    T12 = S11 / S21
    T21 = -S22 / S21
    T22 = 1 / S21

    return np.array([[T11, T12], [T21, T22]], dtype=np.complex128)

def _t_to_s_matrix_conversion(T_matrix):
    """
    Converte uma matriz T de 2x2 para uma matriz S de 2x2.
    USA FÓRMULAS PADRÃO PARA REDES RECÍPROCAS.
    """
    T11 = T_matrix[0, 0]
    T12 = T_matrix[0, 1]
    T21 = T_matrix[1, 0]
    T22 = T_matrix[1, 1]

    if np.abs(T22) < 1e-12:
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)
    
    det_T = T11 * T22 - T12 * T21
    S11 = T12 / T22
    S12 = det_T / T22
    S21 = 1 / T22
    S22 = -T21 / T22

    return np.array([[S11, S12], [S21, S22]], dtype=np.complex128)

def _calculate_total_s_matrix(S_matrix_chrom, chromosome):
    """
    Calcula a matriz S total para um único cromossomo utilizando matrizes de transferência.
    """
    num_frequencies = S_matrix_chrom.shape[2]
    Lambda = chromosome['Lambda']
    
    if Lambda < 1e-12:
        return np.zeros((2, 2, num_frequencies), dtype=np.complex128)
        
    n = int(round(GUIDE_LENGTH / Lambda))
    
    if n <= 0:
        return np.zeros((2, 2, num_frequencies), dtype=np.complex128)
    
    S_total_matrix = np.zeros((2, 2, num_frequencies), dtype=np.complex128)
        
    try:
        for i in range(num_frequencies):
            S_matrix_at_freq = S_matrix_chrom[:, :, i]
            T_matrix_at_freq = _s_to_t_matrix_conversion(S_matrix_at_freq)
            T_total_at_freq = np.linalg.matrix_power(T_matrix_at_freq, n)
            S_total_matrix[:, :, i] = _t_to_s_matrix_conversion(T_total_at_freq)
            
        return S_total_matrix
    except Exception as e:
        print(f"    !!! Erro ao calcular a matriz S total: {e}")
        return np.zeros((2, 2, num_frequencies), dtype=np.complex128)


# --- FUNÇÃO PRINCIPAL MODIFICADA ---
def calculate_mean_S11_for_generation(S_matrixes_for_generation, current_population):
    """
    Calcula o valor MÉDIO de S11 para cada cromossomo de uma geração.

    Args:
        S_matrixes_for_generation (list): Lista de matrizes S (2, 2, num_freq) de cada cromossomo.
        current_population (list): Lista de dicionários com os parâmetros de cada cromossomo.

    Returns:
        Um vetor NumPy 1D [num_cromossomos] com o valor médio da magnitude de S11 para cada um.
    """
    if not S_matrixes_for_generation:
        print("Aviso: Lista de matrizes S está vazia.")
        return np.array([])
        
    # Esta lista irá armazenar um único valor (S11 médio) por cromossomo.
    generation_mean_S11_values = []

    for chrom_id, (S_matrix_chrom, chromosome) in enumerate(zip(S_matrixes_for_generation, current_population)):
        print(f"\nProcessando Cromossomo {chrom_id+1}...")
  
        # 1. Calcula a matriz S do guia completo para todas as frequências
        S_total_matrix = _calculate_total_s_matrix(S_matrix_chrom, chromosome)
        
        # 2. Extrai o espectro completo de S11 (vetor 1D com 500 pontos)
        S11_spectrum_magnitude = np.abs(S_total_matrix[0, 0, :])
        
        # 3. NOVO: Calcula a média de todos os pontos do espectro de S11
        mean_S11 = np.mean(S11_spectrum_magnitude)
        
        # 4. Adiciona o valor médio (um único número) à nossa lista de resultados
        generation_mean_S11_values.append(mean_S11)
        print(f"  -> S11 Médio: {mean_S11:.4f}")

    # 5. Converte a lista de valores médios em um vetor NumPy e retorna
    return np.array(generation_mean_S11_values)