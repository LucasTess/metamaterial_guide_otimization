# lumerical_workflow.py

import lumapi
import os
import numpy as np
import h5py

def run_lumerical_workflow(fdtd, current_s_value, current_w_value, current_l_value,current_height_value,
                            fsp_path, create_lsf_path, run_sim_lsf_path, output_h5_path):

    fdtd.load(fsp_path)
    fdtd.switchtolayout()

    with open(create_lsf_path, 'r') as f:
        create_lsf_content = f.read()
    fdtd.eval(create_lsf_content)

    fdtd.setnamed("Guia Metamaterial", "s", current_s_value)
    fdtd.setnamed("Guia Metamaterial", "w", current_w_value)
    fdtd.setnamed("Guia Metamaterial", "l", current_l_value)
    fdtd.setnamed("Guia Metamaterial", "height", current_height_value)

    #actual_s_in_lumerical = fdtd.getnamed('Guia Metamaterial', 's')
    #print(f"DEBUG: s no Lumerical: {actual_s_in_lumerical:.2e}")

    fdtd.save(fsp_path)

    with open(run_sim_lsf_path, 'r') as f:
        run_sim_lsf_content = f.read()
    fdtd.eval(run_sim_lsf_content)

    monitor_names = ["in"]
    output_results_dir = os.path.join(os.path.dirname(fsp_path), "simulation_spectra")
    os.makedirs(output_results_dir, exist_ok=True)

    simulation_spectra_data = {}
    
    # Obter os vetores de frequência e comprimento de onda do "spectrum" do monitor 'in'
    try:
        # Acessa o dataset 'spectrum' do monitor 'in' e pega o atributo 'f' e 'lambda'
        frequencies_hz = fdtd.getdata("in","f")
    except Exception as e:
        print(f"ERRO CRITICO: Nao foi possivel obter 'f' ou 'lambda' do 'in:spectrum': {e}")
        print("Verifique se o monitor 'in' existe e tem o 'spectrum' calculado.")
        return # Sair da função se não conseguir o domínio de frequência

    simulation_spectra_data['frequencies_hz'] = frequencies_hz
    
    for monitor_name in monitor_names:
        try:
            Ex_complex = fdtd.getdata(f"{monitor_name}","Ex")
            Ey_complex = fdtd.getdata(f"{monitor_name}","Ey")
            Ez_complex = fdtd.getdata(f"{monitor_name}","Ez")

            spectrum_E_magnitude = np.sqrt(np.abs(Ex_complex[0,0,0,:])**2 
                                           + np.abs(Ey_complex[0,0,0,:])**2 
                                           + np.abs(Ez_complex[0,0,0,:])**2)
            
            spectrum_phase = np.angle(Ex_complex[0,0,0,:])
            
            simulation_spectra_data[f'{monitor_name}_spectrum_E_magnitude'] = spectrum_E_magnitude
            simulation_spectra_data[f'{monitor_name}_spectrum_phase'] = spectrum_phase
        except Exception as e:
            print(f"ERRO ao coletar 'spectrum:E' do monitor {monitor_name}: {e}.")
               
    with h5py.File(output_h5_path, 'w') as f:
        for key, value in simulation_spectra_data.items():
            f.create_dataset(key, data=value)
        #f.attrs['s_value'] = current_s_value
    print(f"Espectros salvos em HDF5: {output_h5_path}")