# utils/fitness_functions.py (Atualizado com Média + Desvio Padrão)

import h5py
import numpy as np
from abc import ABC, abstractmethod

# ... (A classe base FitnessStrategy e as outras estratégias permanecem inalteradas) ...
class FitnessStrategy(ABC):
    @abstractmethod
    def calculate(self, output_h5_path: str) -> float:
        pass

class DeltaAmpStrategy(FitnessStrategy):
    # ... (código inalterado) ...
    def calculate(self, output_h5_path: str, monitor_name: str = 'in') -> float:
        # Lógica original do DeltaAmp...
        try:
            with h5py.File(output_h5_path, 'r') as f:
                if 'frequencies_hz' not in f or f'{monitor_name}_spectrum_E_magnitude' not in f:
                    return -np.inf
                frequencies_hz = f['frequencies_hz'][:].flatten()
                spectrum_E_magnitude = f[f'{monitor_name}_spectrum_E_magnitude'][:].flatten()
                peaks, valleys = [], []
                for i in range(1, len(spectrum_E_magnitude) - 1):
                    if spectrum_E_magnitude[i] > spectrum_E_magnitude[i-1] and spectrum_E_magnitude[i] > spectrum_E_magnitude[i+1]:
                        peaks.append((i, spectrum_E_magnitude[i]))
                    elif spectrum_E_magnitude[i] < spectrum_E_magnitude[i-1] and spectrum_E_magnitude[i] < spectrum_E_magnitude[i+1]:
                        valleys.append((i, spectrum_E_magnitude[i]))
                if not peaks or not valleys: return 0.0
                total_delta_amp = 0.0
                for peak_idx, peak_val in peaks:
                    next_valley_val = None
                    for valley_idx, valley_val in valleys:
                        if valley_idx > peak_idx:
                            next_valley_val = valley_val
                            break
                    if next_valley_val is not None:
                        total_delta_amp += abs(peak_val - next_valley_val)
                return total_delta_amp
        except Exception:
            return -np.inf

class BandpassStrategy(FitnessStrategy):
    """
    [LÓGICA DE ENGENHARIA v3] Estratégia multi-objetivo para projetar um
    filtro passa-banda, julgando a transição pelo seu elo mais fraco.
    """
    def __init__(self, f_center: float, bandwidth: float, transition_bandwidth: float,
                 w_rejection: float, w_passband: float, w_transition: float):

        self.f_center = f_center
        self.bandwidth = bandwidth
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
        

    def calculate(self, output_h5_path: str) -> float:
        try:
            with h5py.File(output_h5_path, 'r') as f:
                # ... (carregamento de dados e definição de máscaras inalterados) ...
                frequencies = f['frequencies_hz'][:]
                power_in = f['power_in'][:]
                power_through = f['power_through'][:]
                epsilon = 1e-20
                transmission = np.abs(power_through / (power_in + epsilon))

                f_pass_min = self.f_center - (self.bandwidth / 2)
                f_pass_max = self.f_center + (self.bandwidth / 2)
                f_trans1_min = f_pass_min - self.transition_bandwidth
                f_trans2_max = f_pass_max + self.transition_bandwidth
                
                pass_band_mask = (frequencies >= f_pass_min) & (frequencies <= f_pass_max)
                transition1_mask = (frequencies >= f_trans1_min) & (frequencies < f_pass_min)
                transition2_mask = (frequencies > f_pass_max) & (frequencies <= f_trans2_max)
                non_rejection_mask = (frequencies >= f_trans1_min) & (frequencies <= f_trans2_max)
                stop_band_mask = ~non_rejection_mask
                
                # --- Cálculo dos Scores (com a correção) ---

                # Score de Rejeição 
                T_stop_all = transmission[stop_band_mask]
                score_rej = 1.0 - np.max(T_stop_all) if T_stop_all.size > 0 else 0.0

                # Score da Banda Passante 
                T_pass = transmission[pass_band_mask]
                if T_pass.size == 0:
                    score_pass = 0.0
                else:
                    mean_pass = np.mean(T_pass)
                    std_dev_pass = np.std(T_pass)
                    score_pass = max(0, mean_pass - std_dev_pass)

                #Score de Transição
                T_trans1 = transmission[transition1_mask]
                score_trans1 = max(0, T_trans1[-1] - T_trans1[0]) if T_trans1.size >= 2 else 0.0 # Subida
                
                T_trans2 = transmission[transition2_mask]
                score_trans2 = max(0, T_trans2[0] - T_trans2[-1]) if T_trans2.size >= 2 else 0.0 # Queda
                
                # A qualidade da transição geral é a qualidade da PIOR das duas bordas.
                score_trans = min(score_trans1, score_trans2)

                # --- Fitness Final Ponderado (inalterado) ---
                final_fitness = (self.w_rej * score_rej + 
                                 self.w_pass * score_pass + 
                                 self.w_trans * score_trans)
                
                return float(final_fitness) if not np.isnan(final_fitness) else -np.inf

        except Exception as e:
            print(f"ERRO ao calcular o fitness Bandpass (Engenheiro) para {output_h5_path}: {e}")
            return -np.inf

class LowpassStrategy(FitnessStrategy):
    """
    [LÓGICA DE ENGENHARIA] Estratégia multi-objetivo para projetar um
    filtro passa-baixas, baseada em características de engenharia.
    """
    def __init__(self, f_cutoff: float, transition_bandwidth: float, 
                 w_rejection: float, w_passband: float, w_transition: float):
        
        self.f_cutoff = f_cutoff
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
    
    def calculate(self, output_h5_path: str) -> float:
        try:
            with h5py.File(output_h5_path, 'r') as f:
                frequencies = f['frequencies_hz'][:]
                power_in = f['power_in'][:]
                power_through = f['power_through'][:]
                epsilon = 1e-20
                transmission = np.abs(power_through / (power_in + epsilon))

                # --- 1. Definição das Bandas (INVERTIDAS em relação ao Highpass) ---
                f_min_transition = self.f_cutoff - (self.transition_bandwidth / 2)
                f_max_transition = self.f_cutoff + (self.transition_bandwidth / 2)

                # Banda passante agora é em baixas frequências
                pass_band_mask = frequencies < f_min_transition
                # Banda de rejeição agora é em altas frequências
                stop_band_mask = frequencies > f_max_transition
                transition_mask = (frequencies >= f_min_transition) & (frequencies <= f_max_transition)

                # --- 2. Cálculo dos Scores Individuais Normalizados [0, 1] ---

                # Score de Rejeição (baseado no vazamento máximo)
                T_stop = transmission[stop_band_mask]
                if T_stop.size == 0:
                    score_rej = 0.0
                else:
                    score_rej = 1.0 - np.max(T_stop)

                # Score da Banda Passante (média menos desvio padrão)
                T_pass = transmission[pass_band_mask]
                if T_pass.size == 0:
                    score_pass = 0.0
                else:
                    mean_pass = np.mean(T_pass)
                    std_dev_pass = np.std(T_pass)
                    score_pass = max(0, mean_pass - std_dev_pass)

                # Score de Transição (lógica da "Queda Total")
                T_transition = transmission[transition_mask]
                if T_transition.size < 2:
                    score_trans = 0.0
                else:
                    # Invertido: T_inicial - T_final para recompensar uma queda
                    score_trans = max(0, T_transition[0] - T_transition[-1])

                # --- 3. Fitness Final Ponderado ---
                final_fitness = (self.w_rej * score_rej + 
                                 self.w_pass * score_pass + 
                                 self.w_trans * score_trans)
                
                return float(final_fitness) if not np.isnan(final_fitness) else -np.inf

        except Exception as e:
            print(f"ERRO ao calcular o fitness Lowpass (Engenheiro) para {output_h5_path}: {e}")
            return -np.inf

class HighpassStrategy(FitnessStrategy):
    """
    [LÓGICA DE ENGENHARIA] Estratégia multi-objetivo que avalia o fitness com
    base em três características de engenharia de um filtro:
    1. Score de Rejeição (baseado na pior fuga de sinal).
    2. Score da Banda Passante (baseado na média e planicidade).
    3. Score de Transição (baseado na "Elevação Total").
    O fitness final é uma soma ponderada desses três scores.
    """
    def __init__(self, f_cutoff: float, transition_bandwidth: float, 
                 w_rejection: float, w_passband: float, w_transition: float):
        
        self.f_cutoff = f_cutoff
        self.transition_bandwidth = transition_bandwidth
        self.w_rej = w_rejection
        self.w_pass = w_passband
        self.w_trans = w_transition
    
    def calculate(self, output_h5_path: str) -> float:
        try:
            with h5py.File(output_h5_path, 'r') as f:
                frequencies = f['frequencies_hz'][:]
                power_in = f['power_in'][:]
                power_through = f['power_through'][:]
                epsilon = 1e-20
                transmission = np.abs(power_through / (power_in + epsilon))

                # --- 1. Definição das Bandas ---
                f_min_transition = self.f_cutoff - (self.transition_bandwidth / 2)
                f_max_transition = self.f_cutoff + (self.transition_bandwidth / 2)

                stop_band_mask = frequencies < f_min_transition
                pass_band_mask = frequencies > f_max_transition
                transition_mask = (frequencies >= f_min_transition) & (frequencies <= f_max_transition)

                # --- 2. Cálculo dos Scores Individuais Normalizados [0, 1] ---

                # Score de Rejeição (quão perto de zero está a banda de rejeição)
                T_stop = transmission[stop_band_mask]
                if T_stop.size == 0:
                    score_rej = 0.0
                else:
                    # Baseado no pior ponto (vazamento máximo)
                    score_rej = 1.0 - np.max(T_stop)

                # Score da Banda Passante (quão alta e plana é a banda passante)
                T_pass = transmission[pass_band_mask]
                if T_pass.size == 0:
                    score_pass = 0.0
                else:
                    mean_pass = np.mean(T_pass)
                    std_dev_pass = np.std(T_pass)
                    # Penaliza a falta de planicidade. Não pode ser negativo.
                    score_pass = max(0, mean_pass - std_dev_pass)

                # Score de Transição (quão íngreme e monotônica é a transição)
                T_transition = transmission[transition_mask]
                if T_transition.size < 2:
                    score_trans = 0.0
                else:
                    # Usa a "Elevação Total". Não pode ser negativo.
                    score_trans = max(0, T_transition[-1] - T_transition[0])

                # --- 3. Fitness Final Ponderado ---
                final_fitness = (self.w_rej * score_rej + 
                                 self.w_pass * score_pass + 
                                 self.w_trans * score_trans)
                
                return float(final_fitness) if not np.isnan(final_fitness) else -np.inf

        except Exception as e:
            print(f"ERRO ao calcular o fitness Highpass (Engenheiro) para {output_h5_path}: {e}")
            return -np.inf