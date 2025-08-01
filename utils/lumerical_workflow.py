# lumerical_workflow.py

import lumapi
import os
import h5py
import numpy as np
from .file_handler import remove_file

# ... (prepare_lumerical_job permanece inalterada) ...

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path, geometry_lsf_path,
                                  simulation_lsf_path, simulation_spectra_directory):
    """
    Prepara e executa as simulações para uma geração inteira de cromossomos usando a fila de jobs.
    Após a execução, lê os resultados de cada arquivo FSP, salva em arquivos .h5 e os deleta.
    
    Args:
        # ... (Parâmetros de entrada) ...
        
    Returns:
        Uma lista completa dos caminhos para os arquivos de saída .h5.
    """
    fsp_paths_for_gen = []
    print(f"Preparando e adicionando {len(current_population)} jobs na fila...")
    
    for chromosome in current_population:
        fsp_path = prepare_lumerical_job(
            fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path
        )
        fsp_paths_for_gen.append(fsp_path)
        
        # Adiciona o arquivo FSP com nome único à fila de jobs
        fdtd.addjob(fsp_path)
    
    print("\n  [Job Manager] Executando todos os jobs na fila. Isso pode levar um tempo...")
    fdtd.runjobs()
    print("  [Job Manager] Todos os jobs da geração foram concluídos. Lendo e salvando os resultados...")

    # --- Pós-processamento e salvamento em disco no Python ---
    output_h5_paths = []
    monitor_name = 'in'
    for fsp_path in fsp_paths_for_gen:
        try:
            # Carrega o arquivo FSP já simulado para extrair os dados
            fdtd.load(fsp_path)
            
            # Extrai os dados do monitor 'in'
            Ex_complex = fdtd.getdata(f"{monitor_name}","Ex")
            Ey_complex = fdtd.getdata(f"{monitor_name}","Ey")
            Ez_complex = fdtd.getdata(f"{monitor_name}","Ez")

            E = np.sqrt(np.abs(Ex_complex[0,0,0,:])**2 
                                           + np.abs(Ey_complex[0,0,0,:])**2 
                                           + np.abs(Ez_complex[0,0,0,:])**2)

            f = fdtd.getdata("in","f")
            
            # Define o nome do arquivo H5 com base nos parâmetros do cromossomo
            s_val = fdtd.getnamed("Guia Metamaterial", "s")
            w_val = fdtd.getnamed("Guia Metamaterial", "w")
            l_val = fdtd.getnamed("Guia Metamaterial", "l")
            height_val = fdtd.getnamed("Guia Metamaterial", "height")
            
            h5_file_name = f"spectrum_s{s_val:.2e}_w{w_val:.2e}_l{l_val:.2e}_h{height_val:.2e}.h5"
            h5_path = os.path.join(simulation_spectra_directory, h5_file_name)
            
            # Salva os dados no arquivo H5 usando a biblioteca h5py
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset(f'{monitor_name}_spectrum_E_magnitude', data=E)
                hf.create_dataset(f'frequencies_hz', data=f)
            
            output_h5_paths.append(h5_path)
            
            print(f"  Resultados do cromossomo salvo em: {os.path.basename(h5_path)}")

            # --- AQUI: Limpeza do arquivo FSP apenas se o processamento for bem-sucedido ---
            remove_file(fsp_path)
            remove_file(fsp_path[:-4]+"_p0.log") # Corrigido para .fsp -> _p0.log
            print(f"  [Limpeza] Arquivos temporários removidos para o cromossomo.")

        except Exception as e:
            print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(fsp_path)}: {e}")
            # Os arquivos .fsp e .log não serão removidos em caso de erro, permitindo depuração.

    return output_h5_paths