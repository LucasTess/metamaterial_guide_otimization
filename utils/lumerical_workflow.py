# lumerical_workflow.py (Modificado para coletar Potência com getresult)

import lumapi
import os
import h5py
import numpy as np
import time

def prepare_lumerical_job(fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path, temp_directory):
    """
    Prepara um único arquivo FSP com os parâmetros de um cromossomo e o salva com um nome único.
    """
    # Cria um nome de arquivo FSP único para o cromossomo
    fsp_file_name = f"guide_temp_s{chromosome['s']:.2e}_w{chromosome['w']:.2e}_l{chromosome['l']:.2e}_h{chromosome['height']:.2e}.fsp"
    fsp_path = os.path.join(temp_directory, fsp_file_name)
    
    if not os.path.exists(fsp_base_path):
        raise FileNotFoundError(f"Erro: O arquivo base '{fsp_base_path}' não foi encontrado.")

    # 1. Carrega o arquivo FSP base
    fdtd.load(fsp_base_path)
    fdtd.switchtolayout()

    # 2. Executa o script LSF para criar a geometria
    with open(geometry_lsf_path, 'r') as f:
        create_lsf_content = f.read()
    fdtd.eval(create_lsf_content)

    # 3. Define os parâmetros do cromossomo na geometria
    fdtd.setnamed("Guia Metamaterial", "s", chromosome['s'])
    fdtd.setnamed("Guia Metamaterial", "w", chromosome['w'])
    fdtd.setnamed("Guia Metamaterial", "l", chromosome['l'])
    fdtd.setnamed("Guia Metamaterial", "height", chromosome['height'])

    # 4. Executa o script LSF para adicionar os elementos de simulação
    with open(simulation_lsf_path, 'r') as f:
        simulate_lsf_content = f.read()
    fdtd.eval(simulate_lsf_content)
    
    # 5. Salva o arquivo FSP modificado com um nome único
    fdtd.save(fsp_path)
    
    return fsp_path

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path, geometry_lsf_path,
                                  simulation_lsf_path, simulation_spectra_directory, temp_directory):
    """
    Prepara e executa as simulações para uma geração inteira de cromossomos usando a fila de jobs.
    Após a execução, lê os resultados de cada arquivo FSP e os salva em arquivos .h5.
    """
    fsp_paths_for_gen = []
    print(f"Preparando e adicionando {len(current_population)} jobs na fila...")
    
    for chromosome in current_population:
        fsp_path = prepare_lumerical_job(
            fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path, temp_directory
        )
        fsp_paths_for_gen.append(fsp_path)
        
        # Adiciona o arquivo FSP com nome único à fila de jobs
        fdtd.addjob(fsp_path)
    
    print("\n  [Job Manager] Executando todos os jobs na fila. Isso pode levar um tempo...")
    fdtd.runjobs()

    # Adicionando uma pequena pausa para garantir que os arquivos sejam liberados
    time.sleep(2) 
    
    print("  [Job Manager] Todos os jobs da geração foram concluídos. Lendo e salvando os resultados...")

    # --- Pós-processamento e salvamento em disco no Python ---
    output_h5_paths = []
    for fsp_path in fsp_paths_for_gen:
        try:
            # 1. Carrega o arquivo FSP já simulado para extrair os dados
            fdtd.load(fsp_path)
            
            # --- [MODIFICADO] Extração de dados de potência usando getresult ---
            # Em vez de pegar os campos E e calcular a magnitude, pegamos o resultado de potência (P)
            # que é pré-calculado pelo monitor do tipo 'port'.
            
            # 2. Extrai os dados do port 'in'
            port_in_result = fdtd.getresult("in", "P")
            power_in = port_in_result['P'].flatten()  # Achatamos para garantir 1D
            power_in = np.abs(power_in)
            print("Pontencia in:")
            print(power_in)
            frequencies = port_in_result['f'].flatten()

            # 3. Extrai os dados do port 'through'
            port_through_result = fdtd.getresult("through", "P")
            power_through = port_through_result['P'].flatten() # Achatamos para garantir 1D
            power_through = np.abs(power_through)
            print("Pontencia through")
            print(power_through)           
            # 4. Define o nome do arquivo H5 com base nos parâmetros do cromossomo
            s_val = fdtd.getnamed("Guia Metamaterial", "s")
            w_val = fdtd.getnamed("Guia Metamaterial", "w")
            l_val = fdtd.getnamed("Guia Metamaterial", "l")
            height_val = fdtd.getnamed("Guia Metamaterial", "height")
            
            h5_file_name = f"spectrum_s{s_val:.2e}_w{w_val:.2e}_l{l_val:.2e}_h{height_val:.2e}.h5"
            h5_path = os.path.join(simulation_spectra_directory, h5_file_name)
            
            # --- [MODIFICADO] Salva os novos dados de potência no arquivo H5 ---
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('frequencies_hz', data=frequencies)
                hf.create_dataset('power_in', data=power_in)
                hf.create_dataset('power_through', data=power_through)
            
            output_h5_paths.append(h5_path)
            
            print(f"  Resultados do cromossomo salvo em: {os.path.basename(h5_path)}")

        except Exception as e:
            print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(fsp_path)}: {e}")
            
    return output_h5_paths