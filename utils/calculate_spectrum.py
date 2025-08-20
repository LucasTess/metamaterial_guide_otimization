# calculate_spectrum.py

import numpy as np
import matplotlib.pyplot as plt

# Definimos o comprimento total do guia de onda (constante)
GUIDE_LENGTH = 40e-6

def _create_gaussian_input_spectrum(frequencies):
    """
    Cria um espectro de entrada gaussiano dinamicamente com base no array de frequências.
    A frequência central é o ponto médio do intervalo de frequências.
    A largura (FWHM) é 10% da largura total do intervalo de frequências.

    Args:
        frequencies (array): O array NumPy das frequências da simulação.

    Returns:
        Um array 2D NumPy com o espectro gaussiano no porto 1.
    """
    num_frequencies = len(frequencies)
    
    center_freq = np.mean(frequencies)
    fwhm = (frequencies[-1] - frequencies[0]) * 0.1 
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    gaussian_amplitude = np.exp(-0.5 * ((frequencies - center_freq) / sigma)**2)

    input_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)
    input_spectrum[0, :] = gaussian_amplitude
    
    return input_spectrum

def _s_to_t_matrix_conversion(S_matrix):
    """
    Converte uma matriz S de 2x2 para uma matriz T de 2x2.
    
    Args:
        S_matrix (array): A matriz S de 2x2.
    
    Returns:
        A matriz T de 2x2.
    """
    S11 = S_matrix[0, 0]
    S12 = S_matrix[0, 1]
    S21 = S_matrix[1, 0]
    S22 = S_matrix[1, 1]

    # Evita divisão por zero
    if np.abs(S21) < 1e-12:
        print("Aviso: S21 próximo de zero. A conversão S->T pode ser instável.")
        # Retorna uma matriz T com valores para uma reflexão quase total
        return np.array([[1/S21, S11/S21], [-S22/S21, 1/S21]], dtype=np.complex128)

    T11 = (S11*S22 - S12*S21) / S21
    T12 = S11 / S21
    T21 = -S22 / S21
    T22 = 1 / S21

    return np.array([[T11, T12], [T21, T22]], dtype=np.complex128)

def _t_to_s_matrix_conversion(T_matrix):
    """
    Converte uma matriz T de 2x2 para uma matriz S de 2x2.
    
    Args:
        T_matrix (array): A matriz T de 2x2.
    
    Returns:
        A matriz S de 2x2.
    """
    T11 = T_matrix[0, 0]
    T12 = T_matrix[0, 1]
    T21 = T_matrix[1, 0]
    T22 = T_matrix[1, 1]

    # Evita divisão por zero
    if np.abs(T22) < 1e-12:
        print("Aviso: T22 próximo de zero. A conversão T->S pode ser instável.")
        return np.zeros((2, 2), dtype=np.complex128)
    
    S11 = T12 / T22
    S12 = (T11*T22 - T12*T21) / T22
    S21 = 1 / T22
    S22 = -T21 / T22

    return np.array([[S11, S12], [S21, S22]], dtype=np.complex128)

def _calculate_total_s_matrix(S_matrix_chrom, chromosome):
    """
    Calcula a matriz S total para um único cromossomo utilizando matrizes de transferência.
    
    Args:
        S_matrix_chrom (array): A matriz S 3D de um cromossomo.
        chromosome (dict): Dicionário com os parâmetros do cromossomo.
    
    Returns:
        Um array 3D com a matriz S total, ou um array de zeros se houver erro.
    """
    num_frequencies = S_matrix_chrom.shape[2]
    
    delta = chromosome['l'] + chromosome['s']
    n = int(GUIDE_LENGTH / delta)
    
    print(f"  Calculando S total: delta={delta:.2e}, n={n}")

    if n <= 0:
        print(f"    Aviso: O número de períodos 'n' é zero ou negativo ({n}).")
        return np.zeros((2, 2, num_frequencies), dtype=np.complex128)
    else:
        S_total_matrix = np.zeros((2, 2, num_frequencies), dtype=np.complex128)
        
        try:
            for i in range(num_frequencies):
                S_matrix_at_freq = S_matrix_chrom[:, :, i]

                # Converte S para T com a fórmula CORRIGIDA
                T_matrix_at_freq = _s_to_t_matrix_conversion(S_matrix_at_freq)
                
                # Cascateamento: T_total = T^n
                T_total_at_freq = np.linalg.matrix_power(T_matrix_at_freq, n)
                
                # Converte T_total de volta para S_total com a fórmula CORRIGIDA
                S_total_at_freq = _t_to_s_matrix_conversion(T_total_at_freq)
                
                S_total_matrix[:, :, i] = S_total_at_freq
                
            return S_total_matrix

        except Exception as e:
            print(f"    !!! Erro ao calcular a matriz S total: {e}")
            return np.zeros((2, 2, num_frequencies), dtype=np.complex128)

def calculate_generation_spectra(S_matrixes_for_generation, current_population, frequencies):
    """
    Calcula o espectro de saída para cada cromossomo de uma geração.
    Retorna o espectro refletido na porta 1.

    Args:
        S_matrixes_for_generation (list): Uma lista de arrays NumPy, onde cada um é
                                         a matriz S de um cromossomo.
        current_population (list): Uma lista de dicionários com os parâmetros de cada cromossomo.
        frequencies (array): O array NumPy das frequências da simulação.

    Returns:
        Um array NumPy 2D onde cada linha é o espectro de reflexão de um cromossomo.
    """
    frequencies = frequencies.flatten()
    
    num_chromosomes = len(current_population)
    if num_chromosomes == 0:
        print("Aviso: População vazia. Retornando array vazio.")
        return np.array([])
        
    num_frequencies = S_matrixes_for_generation[0].shape[2]
    all_output_spectra = []

    input_spectrum = _create_gaussian_input_spectrum(frequencies)
    
    for chrom_id, (S_matrix_chrom, chromosome) in enumerate(zip(S_matrixes_for_generation, current_population)):
        
        # --- PASSO DE DEBUG: IMPRIMINDO VALORES BRUTOS DA MATRIZ S DA CÉLULA UNITÁRIA ---
        center_freq_index = num_frequencies // 2
        print("\n--- Verificação de Dados (Debugging) ---")
        print(f"Cromossomo {chrom_id+1}:")
        print(f"  Frequência de Inspecao: {frequencies[center_freq_index]:.4e} Hz")
        print(f"  Matriz S da Célula Unitária (para esta frequencia):")
        print(S_matrix_chrom[:, :, center_freq_index])
        print("------------------------------------------")

        # Plotar o espectro de reflexão da CÉLULA UNITÁRIA
        reflection_spectrum_unit_cell = S_matrix_chrom[0, 0, :] * input_spectrum[0, :]
        plot_spectrum(reflection_spectrum_unit_cell, frequencies, title=f"Espectro de Reflexão da Célula Unitária (Cromossomo {chrom_id+1})")
        
        # Plotar o espectro de transmissão da CÉLULA UNITÁRIA
        transmission_spectrum_unit_cell = S_matrix_chrom[1, 0, :] * input_spectrum[0, :]
        plot_spectrum(transmission_spectrum_unit_cell, frequencies, title=f"Espectro de Transmissão da Célula Unitária (Cromossomo {chrom_id+1})")
        
        # 2. Calcular o espectro do GUIA COMPLETO e plotá-lo
        S_total_matrix = _calculate_total_s_matrix(S_matrix_chrom, chromosome)

        # --- NOVO PASSO DE DEBUG: IMPRIMINDO A MATRIZ S DO GUIA COMPLETO ---
        print("\n--- Verificação de Dados (Debugging) ---")
        print(f"  Matriz S do Guia Completo (para a mesma frequencia):")
        print(S_total_matrix[:, :, center_freq_index])
        print("------------------------------------------")
        
        output_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)
        for i in range(num_frequencies):
            output_spectrum[:, i] = S_total_matrix[:, :, i] @ input_spectrum[:, i]

        output_spectrum_port_1 = output_spectrum[0, :]
        plot_spectrum(output_spectrum_port_1, frequencies, title=f"Espectro de Reflexão do Guia Completo (Cromossomo {chrom_id+1})")
        
        all_output_spectra.append(output_spectrum_port_1)

    return np.vstack(all_output_spectra)


def plot_spectrum(spectrum, frequencies, title="Espectro de Reflexão", save_path=None):
    """
    Plota a magnitude de um espectro complexo.

    Args:
        spectrum (array): O array com os valores complexos do espectro (1D).
        frequencies (array): O array com os valores de frequência (1D).
        title (str): O título do gráfico.
        save_path (str): O caminho para salvar o gráfico. Se for None, o gráfico será exibido.
    """
    if spectrum.ndim > 1:
        print("Aviso: A função de plotagem espera um array 1D. Plotando a primeira linha.")
        spectrum = spectrum[0]

    magnitude = np.abs(spectrum)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies.flatten(), magnitude)
    plt.title(title)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude do Espectro (Reflexão)")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()