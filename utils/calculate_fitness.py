# utils/calculate_fitness.py

import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Constante da velocidade da luz no vácuo (m/s)
C_LIGHT = 299792458.0

def generate_gaussian_spectrum(frequencies, center_frequency, fwhm, max_amplitude=3.0):
    """
    Gera um espectro de pulso gaussiano (sinal de entrada).
    """
    if fwhm < 1e-12:
        sigma = 1e-12
    else:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    exponent = -((frequencies - center_frequency)**2) / (2 * sigma**2)
    return max_amplitude * np.exp(exponent)

def _calculate_single_delta_amp(reflected_spectrum):
    """
    Calcula o fitness 'delta_amp' para um único espectro de amplitude.
    """
    peak_indices, _ = find_peaks(reflected_spectrum, height=0.05, distance=5)
    valley_indices, _ = find_peaks(-reflected_spectrum, distance=5)
    
    if len(peak_indices) < 1 or len(valley_indices) < 1:
        return 0.0

    extrema_indices = np.sort(np.concatenate([peak_indices, valley_indices]))
    extrema_amplitudes = reflected_spectrum[extrema_indices]
    return np.sum(np.abs(np.diff(extrema_amplitudes)))

def calculate_fitness_for_generation(
    total_S_matrices, 
    frequencies, 
    fwhm_fraction=0.75,
    plot_best_spectrum=False, 
    generation_num=None,
    output_directory="."
):
    """
    Recebe as matrizes S totais e calcula o fitness delta_amp para cada cromossomo.
    """
    frequencies = frequencies.flatten()
    
    if total_S_matrices.ndim != 4 or total_S_matrices.shape[1:3] != (2, 2):
        raise ValueError("O array de entrada 'total_S_matrices' deve ter o formato [N, 2, 2, F].")
        
    gaussian_center_freq = np.mean(frequencies)
    total_bandwidth = frequencies[-1] - frequencies[0]
    gaussian_fwhm = total_bandwidth * fwhm_fraction
    input_spectrum = generate_gaussian_spectrum(frequencies, gaussian_center_freq, gaussian_fwhm)
    
    s11_complex_spectrums = total_S_matrices[:, 0, 0, :]
    s11_magnitude_spectrums = np.abs(s11_complex_spectrums)
    
    reflected_spectrums = s11_magnitude_spectrums * input_spectrum
    
    fitness_values = np.apply_along_axis(_calculate_single_delta_amp, axis=1, arr=reflected_spectrums)
    
    if plot_best_spectrum and generation_num is not None:
            try:
                # 1. Encontra os dados do melhor cromossomo da geração
                best_chrom_index = np.argmax(fitness_values)
                best_s11_spectrum = s11_magnitude_spectrums[best_chrom_index]
                best_reflected_spectrum = reflected_spectrums[best_chrom_index]
                best_fitness = fitness_values[best_chrom_index]
                wavelengths_nm = (C_LIGHT / frequencies) * 1e9
                
                # --- GRÁFICO 1: Espectro de Reflexão |S11| (Resposta Intrínseca) ---
                plt.figure(figsize=(12, 7))
                plt.plot(wavelengths_nm, best_s11_spectrum, color='purple', linewidth=2, label='Reflexão |S11| (Melhor Cromossomo)')
                plt.title(f'Resposta Intrínseca |S11| - Melhor Cromossomo (Geração {generation_num})', fontsize=16)
                plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
                plt.ylabel("Magnitude |S11|", fontsize=12)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim(0, 1.1)
                plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())
                
                s11_plot_path = os.path.join(output_directory, "realtime_S11_spectrum.png")
                plt.savefig(s11_plot_path, dpi=150)
                plt.close()

                # --- GRÁFICO 2: Espectro Refletido Final (Base para o Fitness) ---
                peaks, _ = find_peaks(best_reflected_spectrum, height=0.05, distance=5)
                valleys, _ = find_peaks(-best_reflected_spectrum, distance=5)

                plt.figure(figsize=(12, 7))
                plt.plot(wavelengths_nm, input_spectrum, 'k--', alpha=0.6, label='Pulso de Entrada (Gaussiano)')
                plt.plot(wavelengths_nm, best_reflected_spectrum, color='blue', linewidth=2, label='Espectro Refletido Final')
                plt.plot(wavelengths_nm[peaks], best_reflected_spectrum[peaks], "x", color='red', markersize=8, mew=2, label='Picos Detectados')
                plt.plot(wavelengths_nm[valleys], best_reflected_spectrum[valleys], "o", color='green', markersize=7, label='Vales Detectados')

                title = f"Espectro Refletido Final - Melhor Cromossomo (Geração {generation_num})\nFitness (delta_amp) = {best_fitness:.4f}"
                plt.title(title, fontsize=16)
                plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
                plt.ylabel("Amplitude Refletida", fontsize=12)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim(bottom=0)
                plt.xlim(wavelengths_nm.min(), wavelengths_nm.max())

                reflected_plot_path = os.path.join(output_directory, "realtime_reflected_spectrum.png")
                plt.savefig(reflected_plot_path, dpi=150)
                plt.close()
                
                print(f"  [Debug Plots] Gráficos de diagnóstico salvos em: {output_directory}")

            except Exception as e:
                print(f"  [Debug Plot] !!! Erro ao gerar gráficos de diagnóstico: {e}")
            
            
    return fitness_values