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
    [LÓGICA DE CURVA-ALVO] Estratégia que calcula o fitness com base na
    similaridade entre a curva de transmissão simulada e uma curva
    sigmoide ideal (filtro passa-altas).
    """
    def __init__(self, frequencies: np.ndarray, f_cutoff: float, max_transmission: float, steepness: float):
        """
        Inicializa a estratégia e pré-calcula a curva-alvo.

        Args:
            frequencies (np.ndarray): O vetor de frequências da simulação.
            f_cutoff (float): A frequência de corte do filtro alvo.
            max_transmission (float): O nível de transmissão máximo desejado.
            steepness (float): Um fator que controla a inclinação da transição.
        """
        self.target_curve = HighpassStrategy._create_target_sigmoid(
            frequencies, f_cutoff, max_transmission, steepness
        )
        if self.target_curve.size == 0:
            raise ValueError("A curva alvo (target_curve) não pode ser vazia.")

    @staticmethod
    def _create_target_sigmoid(frequencies: np.ndarray, f_cutoff: float, max_transmission: float, steepness: float) -> np.ndarray:
        """ Gera a curva-alvo em forma de sigmoide. """
        # A constante no expoente ajuda a normalizar o 'steepness'
        k = steepness / (frequencies[-1] - frequencies[0])
        return max_transmission / (1 + np.exp(-k * (frequencies - f_cutoff)))

    def calculate(self, output_h5_path: str) -> float:
        """
        Calcula o fitness como 1 / (1 + MSE), onde MSE é o erro quadrático
        médio entre a curva simulada e a curva alvo.
        """
        try:
            with h5py.File(output_h5_path, 'r') as f:
                power_in = f['power_in'][:]
                power_through = f['power_through'][:]

                epsilon = 1e-20
                transmission = np.abs(power_through / (power_in + epsilon))

                if len(transmission) != len(self.target_curve):
                    print(f"AVISO: Incompatibilidade de tamanho. T_simulado: {len(transmission)}, T_alvo: {len(self.target_curve)}")
                    return -np.inf

                # Calcula o Erro Quadrático Médio (Mean Squared Error)
                mse = np.mean((transmission - self.target_curve)**2)

                # Converte o erro em um score de fitness (valor máximo = 1)
                fitness_score = 1.0 / (1.0 + mse)

                return float(fitness_score) if not np.isnan(fitness_score) else -np.inf

        except Exception as e:
            print(f"ERRO ao calcular o fitness Highpass (TargetCurve) para {output_h5_path}: {e}")
            return -np.inf