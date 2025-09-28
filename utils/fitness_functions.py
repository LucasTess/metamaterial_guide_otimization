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
    # ... (código inalterado) ...
    def __init__(self, center_freq: float, bandwidth: float):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
    def calculate(self, output_h5_path: str) -> float:
        return 0.0

class LowpassStrategy(FitnessStrategy):
    # ... (código inalterado) ...
    def __init__(self, cutoff_freq: float):
        self.cutoff_freq = cutoff_freq
    def calculate(self, output_h5_path: str) -> float:
        return 0.0


class HighpassStrategy(FitnessStrategy):
    """
    [IMPLEMENTADO - LÓGICA ROBUSTA v2] Estratégia para otimizar um filtro passa-altas.
    Usa a MÉDIA para avaliar o desempenho geral da banda e penaliza a falta de
    planicidade (alto desvio padrão), conforme a discussão.
    """
    def __init__(self, cutoff_freq: float):
        self.cutoff_freq = cutoff_freq
        if cutoff_freq <= 0:
            raise ValueError("A frequência de corte (cutoff_freq) deve ser um valor positivo.")

    def calculate(self, output_h5_path: str) -> float:
        try:
            with h5py.File(output_h5_path, 'r') as f:
                frequencies = f['frequencies_hz'][:]
                power_in = f['power_in'][:]
                power_through = f['power_through'][:]

                epsilon = 1e-20
                transmission = np.abs(power_through / (power_in + epsilon))

                stop_band_mask = frequencies <= self.cutoff_freq
                pass_band_mask = frequencies > self.cutoff_freq
                
                if not np.any(stop_band_mask) or not np.any(pass_band_mask):
                    print(f"AVISO: A frequência de corte {self.cutoff_freq/1e12:.2f} THz está fora do range da simulação. O fitness será 0.")
                    return 0.0
                
                T_pass = transmission[pass_band_mask]
                T_stop = transmission[stop_band_mask]

                # --- [LÓGICA ATUALIZADA CONFORME SUA SUGESTÃO] ---
                
                # 1. Calcula a MÉDIA de cada banda (para avaliar o desempenho homogêneo)
                mean_pass = np.mean(T_pass)
                mean_stop = np.mean(T_stop)

                # 2. Calcula o DESVIO PADRÃO de cada banda como uma métrica de "não-planicidade"
                std_dev_pass = np.std(T_pass)
                std_dev_stop = np.std(T_stop)

                # 3. A pontuação de contraste é a diferença das médias
                contrast_score = mean_pass - mean_stop

                # 4. A penalidade é a soma dos desvios padrão
                flatness_penalty = std_dev_pass + std_dev_stop

                # 5. O Fitness final recompensa o contraste e penaliza a falta de planicidade
                fitness_score = contrast_score - flatness_penalty
                
                # --- Fim da Lógica Atualizada ---
                
                return float(fitness_score) if not np.isnan(fitness_score) else -np.inf

        except Exception as e:
            print(f"ERRO ao calcular o fitness Highpass para {output_h5_path}: {e}")
            return -np.inf