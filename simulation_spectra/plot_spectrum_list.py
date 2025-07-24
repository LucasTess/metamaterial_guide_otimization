import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def plot_h5_spectra(simulation_spectra_dir, s_values_range=None):
    h5_files_data = []

    # Regex para extrair o valor de 's' do nome do arquivo
    s_pattern = re.compile(r"spectrum_time_monitors_s_([0-9eE\-\+\.]+)\.h5")

    for filename in os.listdir(simulation_spectra_dir):
        if filename.endswith(".h5"):
            match = s_pattern.match(filename)
            if match:
                s_str = match.group(1).replace('e+', 'e')
                s_value = float(s_str)

                if s_values_range and not (s_values_range[0] <= s_value <= s_values_range[1]):
                    continue

                file_path = os.path.join(simulation_spectra_dir, filename)
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'frequencies_hz' in f and \
                           'in_spectrum_E_magnitude' in f and \
                           'through_spectrum_E_magnitude' in f:
                            
                            frequencies_hz = f['frequencies_hz'][:].flatten()
                            in_E_magnitude = f['in_spectrum_E_magnitude'][:].flatten()
                            through_E_magnitude = f['through_spectrum_E_magnitude'][:].flatten()
                            delta_value = f.attrs.get('delta_value', s_value)

                            h5_files_data.append({
                                's_value': s_value,
                                'delta_value': delta_value,
                                'frequencies_hz': frequencies_hz,
                                'wavelength_nm': (299792458.0 / frequencies_hz) * 1e9,
                                'in_E': in_E_magnitude,
                                'through_E': through_E_magnitude
                            })
                        else:
                            print(f"AVISO: Arquivo {filename} não contém todos os datasets esperados.")
                except Exception as e:
                    print(f"ERRO ao ler {filename}: {e}")

    h5_files_data.sort(key=lambda x: x['s_value'])

    if not h5_files_data:
        print(f"Nenhum arquivo .h5 encontrado ou processado no diretório: {simulation_spectra_dir}")
        if s_values_range:
            print(f"Verifique se há arquivos .h5 com 's' entre {s_values_range[0]:.2e} e {s_values_range[1]:.2e}.")
        return

    # Calcular os limites globais para o eixo Y
    all_in_E_magnitudes = np.concatenate([data['in_E'] for data in h5_files_data])
    all_through_E_magnitudes = np.concatenate([data['through_E'] for data in h5_files_data])
    
    # Combinar todas as magnitudes para encontrar o min/max global
    all_magnitudes = np.concatenate((all_in_E_magnitudes, all_through_E_magnitudes))
    
    # Ignorar NaNs e Infs ao calcular min/max
    y_min_global = np.nanmin(all_magnitudes[np.isfinite(all_magnitudes)])
    y_max_global = np.nanmax(all_magnitudes[np.isfinite(all_magnitudes)])

    # Adicionar uma pequena margem para melhor visualização
    y_range = y_max_global - y_min_global
    y_min_global -= y_range * 0.05
    y_max_global += y_range * 0.05

    # Definir os tamanhos dos lotes (chunks)
    chunk_sizes = [3, 3, 3, 2]

    current_idx = 0
    for chunk_num, chunk_size in enumerate(chunk_sizes):
        chunk_data = h5_files_data[current_idx : current_idx + chunk_size]
        if not chunk_data:
            break

        num_rows_in_chunk = len(chunk_data)
        if num_rows_in_chunk == 0:
            continue

        fig, axes = plt.subplots(num_rows_in_chunk, 2, figsize=(14, 5 * num_rows_in_chunk), sharex='col', sharey='row')

        if num_rows_in_chunk == 1:
            axes = np.array([axes])

        for i, data in enumerate(chunk_data):
            s_val_formatted = f"{data['s_value']:.2e}"
            label_text = f"Delta={data['delta_value']:.2e} m" if not np.isnan(data['delta_value']) else f"s={s_val_formatted} m"

            # Coluna da esquerda: Espectro 'in'
            axes[i, 0].plot(data['wavelength_nm'], data['in_E'], label=label_text)
            axes[i, 0].set_title(f"Monitor 'in' (s={s_val_formatted} m)")
            axes[i, 0].set_ylabel("Magnitude do Campo Elétrico (E)")
            axes[i, 0].grid(True)
            axes[i, 0].legend()
            axes[i, 0].set_ylim(y_min_global, y_max_global) # Aplicar escala Y global
            # Definir marcações do eixo Y para ser de 1 em 1
            axes[i, 0].set_yticks(np.arange(np.floor(y_min_global), np.ceil(y_max_global) + 1, 1))


            # Coluna da direita: Espectro 'through'
            axes[i, 1].plot(data['wavelength_nm'], data['through_E'], label=label_text, color='orange')
            axes[i, 1].set_title(f"Monitor 'through' (Delta={s_val_formatted} m)")
            axes[i, 1].grid(True)
            axes[i, 1].legend()
            axes[i, 1].set_ylim(y_min_global, y_max_global) # Aplicar escala Y global
            # Definir marcações do eixo Y para ser de 1 em 1
            axes[i, 1].set_yticks(np.arange(np.floor(y_min_global), np.ceil(y_max_global) + 1, 1))


        fig.text(0.5, 0.04, 'Comprimento de Onda (nm)', ha='center', va='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.suptitle(f"Espectros dos Monitores 'in' e 'through' - Lote {chunk_num + 1}", y=1.02, fontsize=16)
        plt.show()

        current_idx += chunk_size

if __name__ == "__main__":
    project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\metamaterial_guide_otimization"
    simulation_spectra_dir = os.path.join(project_directory, "simulation_spectra")

    s_range = (0.0, 5.0e-6)

    plot_h5_spectra(simulation_spectra_dir, s_values_range=s_range)