import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_single_h5_spectrum(file_path, monitor_name='in'):
    """
    Lê um arquivo .h5, extrai os dados de espectro do monitor 'in',
    converte frequências para comprimento de onda e plota o resultado.

    Args:
        file_path (str): Caminho completo para o arquivo .h5 a ser lido.
        monitor_name (str): Nome do monitor cujos dados de espectro serão plotados.
                            Por padrão, é 'in'.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Verifica se os datasets esperados existem
            freq_dataset_name = 'frequencies_hz'
            spectrum_dataset_name = f'{monitor_name}_spectrum_E_magnitude'

            if freq_dataset_name not in f or spectrum_dataset_name not in f:
                print(f"Erro: Arquivo '{file_path}' não contém os datasets esperados.")
                print(f"Esperados: '{freq_dataset_name}' e '{spectrum_dataset_name}'.")
                print(f"Datasets encontrados: {list(f.keys())}")
                return

            # Carrega os dados
            frequencies_hz = f[freq_dataset_name][:]
            # A magnitude do campo elétrico do monitor 'in'
            in_E_magnitude = f[spectrum_dataset_name][:]

            # Garante que os dados sejam 1D (achata se forem multidimensionais)
            frequencies_hz = frequencies_hz.flatten()
            in_E_magnitude = in_E_magnitude.flatten()

            # Constante da velocidade da luz no vácuo
            c = 299792458.0 # m/s

            # Converte frequência (Hz) para comprimento de onda (nm)
            # lambda = c / f, e 1 metro = 1e9 nanometros
            wavelength_nm = (c / frequencies_hz) * 1e9

            # Cria o gráfico
            plt.figure(figsize=(10, 6))
            plt.plot(wavelength_nm, in_E_magnitude, label=f'Monitor \'{monitor_name}\' Espectro E-Magnitude')
            plt.title(f'Espectro do Monitor \'{monitor_name}\' para {os.path.basename(file_path)}')
            plt.xlabel('Comprimento de Onda (nm)')
            plt.ylabel('Magnitude do Campo Elétrico (E)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{file_path}'.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo '{file_path}': {e}")

if __name__ == "__main__":
    # Defina o diretório base do seu projeto
    _project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\\metamaterial_guide_otimization"
    # Defina o subdiretório onde os arquivos .h5 são salvos
    _simulation_spectra_directory = os.path.join(_project_directory, "simulation_spectra")
    chosen_file =os.path.join(_simulation_spectra_directory, 'current_monitor_data.h5') 
    
    if chosen_file:
        print(f"Tentando plotar o arquivo: {chosen_file}")
        plot_single_h5_spectrum(chosen_file, monitor_name='in')
    else:
        print(f"Nenhum arquivo .h5 do monitor 'in' encontrado no diretório: {_simulation_spectra_directory}")
        print("Por favor, certifique-se de que o diretório contém arquivos .h5 com o padrão 'in_monitor_'.")
        print("Certifique-se também de que já rodou uma simulação para gerar esses arquivos.")