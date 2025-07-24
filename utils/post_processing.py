import h5py
import numpy as np
import os
import re

def calculate_delta_amp(output_h5_path, monitor_name='in'):
    with h5py.File(output_h5_path, 'r') as f:
        # Verifica se o monitor existe e tem os datasets esperados
        if 'frequencies_hz' not in f or \
           f'{monitor_name}_spectrum_E_magnitude' not in f:
            raise ValueError(f"Arquivo H5 '{output_h5_path}' não contém os datasets esperados ('frequencies_hz' e '{monitor_name}_spectrum_E_magnitude').")

        # Coleta os dados de frequência e magnitude do campo elétrico
        frequencies_hz = f['frequencies_hz'][:].flatten()
        spectrum_E_magnitude = f[f'{monitor_name}_spectrum_E_magnitude'][:].flatten()

        # Encontra picos e vales
        # Usamos uma detecção de picos/vales simples: um ponto é pico se maior que vizinhos, vale se menor.
        peaks = []
        valleys = []

        # Itera sobre o espectro para encontrar picos e vales
        for i in range(1, len(spectrum_E_magnitude) - 1):
            if spectrum_E_magnitude[i] > spectrum_E_magnitude[i-1] and spectrum_E_magnitude[i] > spectrum_E_magnitude[i+1]:
                peaks.append((i, spectrum_E_magnitude[i]))
            elif spectrum_E_magnitude[i] < spectrum_E_magnitude[i-1] and spectrum_E_magnitude[i] < spectrum_E_magnitude[i+1]:
                valleys.append((i, spectrum_E_magnitude[i]))
        
        total_delta_amp = 0.0
        # Itera sobre os picos e encontra o vale imediatamente seguinte
        for peak_idx, peak_val in peaks:
            # Procura o primeiro vale que sucede o pico atual
            next_valley_val = None
            for valley_idx, valley_val in valleys:
                if valley_idx > peak_idx:
                    next_valley_val = valley_val
                    break
            
            if next_valley_val is not None:
                total_delta_amp += abs(peak_val - next_valley_val) # Acumula a diferença absoluta

        return total_delta_amp

# if __name__ == '__main__':
#     # --- Exemplo de Uso (para teste local e modularidade) ---
#     _project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\\metamaterial_guide_otimization"
#     _spectra_directory = os.path.join(_project_directory, "simulation_spectra")

#     # Exemplo de nome de arquivo para um cromossomo hipotético
#     # O main.py passará o 'simulation_id' para gerar o nome do arquivo.
#     _test_simulation_id = "s_1.50e-07" 

#     output_h5_path = "in_monitor_crom_gen2_chrom2.h5"
#     output_h5_path = os.path.join(_spectra_directory, output_h5_path)
#     delta_amp_value = calculate_delta_amp(output_h5_path, monitor_name='in')
#     print(f"Delta Amplitude Acumulada para o arquivo de teste ({output_h5_path}): {delta_amp_value:.4f}")
