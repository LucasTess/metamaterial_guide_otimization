# calculate_spectrum.py

import numpy as np

# Definimos o comprimento total do guia de onda (constante)
GUIDE_LENGTH = 40e-6

def calculate_generation_spectra(S_matrixes_for_generation, current_population):
    """
    Calcula o espectro de saída para cada cromossomo de uma geração.
    Retorna o espectro refletido na porta 1.

    Args:
        S_matrixes_for_generation (list): Uma lista de arrays NumPy, onde cada um é
                                         a matriz S de um cromossomo. As dimensões esperadas são (2, 2, n_freq).
        current_population (list): Uma lista de dicionários com os parâmetros de cada cromossomo.

    Returns:
        Um array NumPy 2D onde cada linha é o espectro de reflexão de um cromossomo.
        As dimensões são (num_cromossomos, num_pontos_de_frequencia).
    """

    num_chromosomes = len(current_population)
    if num_chromosomes == 0:
        print("Aviso: População vazia. Retornando array vazio.")
        return np.array([])
        
    num_frequencies = S_matrixes_for_generation[0].shape[2]
    all_output_spectra = []

    # Cria um espectro de entrada de referência
    input_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)
    input_spectrum[0, :] = 1.0  # Amplitude de 1.0 no porto 1

    for chrom_id, (S_matrix_chrom, chromosome) in enumerate(zip(S_matrixes_for_generation, current_population)):
        
        delta = chromosome['l'] + chromosome['s']
        n = int(GUIDE_LENGTH / delta)
        
        print(f"  Calculando espectro para o cromossomo {chrom_id+1}: delta={delta:.2e}, n={n}")

        if n <= 0:
            print(f"    Aviso: O número de períodos 'n' é zero ou negativo ({n}). O espectro de saída será zero.")
            output_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)
        else:
            try:
                S_total_matrix = np.zeros((2, 2, num_frequencies), dtype=np.complex128)
                output_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)

                for i in range(num_frequencies):
                    S_matrix_at_freq = S_matrix_chrom[:, :, i]
                    S_total_at_freq = np.linalg.matrix_power(S_matrix_at_freq, n)
                    
                    S_total_matrix[:, :, i] = S_total_at_freq
                    output_spectrum[:, i] = S_total_at_freq @ input_spectrum[:, i]

            except Exception as e:
                print(f"    !!! Erro ao calcular a matriz S total ou espectro: {e}")
                output_spectrum = np.zeros((2, num_frequencies), dtype=np.complex128)

        # A única mudança está aqui: agora extraímos o espectro do porto 1 (reflexão)
        output_spectrum_port_1 = output_spectrum[0, :]

        all_output_spectra.append(output_spectrum_port_1)

    return np.vstack(all_output_spectra)