# lumerical_workflow.py (Versão Final Corrigida para ordem de dados Px,Py,Pz)

import lumapi
import os
import h5py
import numpy as np
import time

def prepare_lumerical_job(fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path, temp_directory):
    # ... (esta função permanece inalterada) ...
    fsp_file_name = (f"guide_temp_s{chromosome['s']:.2e}_w{chromosome['w']:.2e}_"
                     f"l{chromosome['l']:.2e}_h{chromosome['height']:.2e}_"
                     f"len{chromosome['total_length']:.2e}.fsp")
    fsp_path = os.path.join(temp_directory, fsp_file_name)
    
    if not os.path.exists(fsp_base_path):
        raise FileNotFoundError(f"Erro: O arquivo base '{fsp_base_path}' não foi encontrado.")

    fdtd.load(fsp_base_path)
    fdtd.switchtolayout()

    with open(geometry_lsf_path, 'r') as f:
        create_lsf_content = f.read()
    fdtd.eval(create_lsf_content)

    fdtd.setnamed("Guia Metamaterial", "s", chromosome['s'])
    fdtd.setnamed("Guia Metamaterial", "w", chromosome['w'])
    fdtd.setnamed("Guia Metamaterial", "l", chromosome['l'])
    fdtd.setnamed("Guia Metamaterial", "height", chromosome['height'])
    fdtd.setnamed("Guia Metamaterial", "total_length", chromosome['total_length'])

    with open(simulation_lsf_path, 'r') as f:
        simulate_lsf_content = f.read()
    fdtd.eval(simulate_lsf_content)
    
    fdtd.save(fsp_path)
    
    return fsp_path

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path, geometry_lsf_path,
                                  simulation_lsf_path, simulation_spectra_directory, temp_directory):
    # ... (a primeira parte da função permanece inalterada) ...
    fsp_paths_for_gen = []
    print(f"Preparando e adicionando {len(current_population)} jobs na fila...")
    
    for chromosome in current_population:
        fsp_path = prepare_lumerical_job(
            fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path, temp_directory
        )
        fsp_paths_for_gen.append(fsp_path)
        fdtd.addjob(fsp_path)
    
    print("\n  [Job Manager] Executando todos os jobs na fila. Isso pode levar um tempo...")
    fdtd.runjobs()

    time.sleep(2) 
    
    print("  [Job Manager] Todos os jobs da geração foram concluídos. Lendo e salvando os resultados...")

    # --- [MODIFICADO] Lógica de pós-processamento robusta a falhas ---
    output_results = [] # Esta lista conterá caminhos de arquivo ou None para falhas
    for fsp_path in fsp_paths_for_gen:
        try:
            fdtd.load(fsp_path)
            
            port_in_result = fdtd.getresult("in", "P")
            raw_power_in = port_in_result['P'].flatten()
            frequencies = port_in_result['f'].flatten()
            num_freq_points = len(frequencies)

            reshaped_power_in = raw_power_in.reshape(num_freq_points, 3)
            power_in = np.abs(reshaped_power_in[:, 0])

            port_through_result = fdtd.getresult("through", "P")
            raw_power_through = port_through_result['P'].flatten()
            reshaped_power_through = raw_power_through.reshape(num_freq_points, 3)
            power_through = np.abs(reshaped_power_through[:, 0])
            
            s_val = fdtd.getnamed("Guia Metamaterial", "s")
            w_val = fdtd.getnamed("Guia Metamaterial", "w")
            l_val = fdtd.getnamed("Guia Metamaterial", "l")
            height_val = fdtd.getnamed("Guia Metamaterial", "height")
            length_val = fdtd.getnamed("Guia Metamaterial", "total_length")
            
            h5_file_name = (f"spectrum_s{s_val:.2e}_w{w_val:.2e}_l{l_val:.2e}_"
                            f"h{height_val:.2e}_len{length_val:.2e}.h5")
            h5_path = os.path.join(simulation_spectra_directory, h5_file_name)
            
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('frequencies_hz', data=frequencies)
                hf.create_dataset('power_in', data=power_in)
                hf.create_dataset('power_through', data=power_through)
            
            # Adiciona o caminho do arquivo de sucesso à lista de resultados
            output_results.append(h5_path)
            
            print(f"  Resultados do cromossomo salvo em: {os.path.basename(h5_path)}")

        except Exception as e:
            print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(fsp_path)}: {e}")
            # Adiciona None à lista de resultados para marcar a falha
            output_results.append(None)
            
    # A função agora retorna uma lista com tamanho igual ao da população
    return output_results