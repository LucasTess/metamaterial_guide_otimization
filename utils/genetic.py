import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations, # 'generations' já está ok
                 s_range, w_range, l_range, height_range):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations # Nome da variável para o número máximo de gerações
        self.param_ranges = {
            's': s_range,
            'w': w_range,
            'l': l_range,
            'height': height_range
        }
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = [] # <--- NOVO: Inicializa o histórico de fitness

        self.reference_params = {
            's': 0.15e-6,
            'w': 0.5e-6,
            'l': 0.15e-6,
            'height': 0.22e-6
        }
        self.initial_mutation_amplitude = {
            's': 0.5 * self.reference_params['s'],
            'w': 0.5 * self.reference_params['w'],
            'l': 0.5 * self.reference_params['l'],
            'height': 0.5 * self.reference_params['height']
        }
        # Define a amplitude da mutação "local" como uma porcentagem do range total
        self.local_mutation_step = {
            param: (self.param_ranges[param][1] - self.param_ranges[param][0]) * 0.05
            for param in self.param_ranges
        }


    def _constrain_param(self, param_name, value):
        min_val, max_val = self.param_ranges[param_name]
        return max(min_val, min(max_val, value))


    def create_chromosome(self, reference_based=False):
        chromosome = {}
        if reference_based:
            for param, ref_val in self.reference_params.items():
                variation_amplitude = self.initial_mutation_amplitude[param]
                val = random.uniform(ref_val - variation_amplitude, ref_val + variation_amplitude)
                chromosome[param] = self._constrain_param(param, val)
        else:
            for param in self.param_ranges:
                chromosome[param] = random.uniform(*self.param_ranges[param])
        return chromosome


    def initialize_population(self):
        self.population = []
        num_ref_based = self.population_size // 2  # Metade da população baseada em referência
        num_random = self.population_size - num_ref_based # A outra metade aleatória

        for _ in range(num_ref_based):
            self.population.append(self.create_chromosome(reference_based=True))

        for _ in range(num_random):
            self.population.append(self.create_chromosome(reference_based=False))

        random.shuffle(self.population)


    def calculate_fitness(self, delta_amp):
        if np.isinf(delta_amp) or np.isnan(delta_amp):
            return -float('inf')
        return delta_amp


    def select_parents(self):
        pool = random.sample(self.population, min(5, len(self.population)))
        parent1 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        pool = random.sample(self.population, min(5, len(self.population)))
        parent2 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        return parent1, parent2


    def crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        keys = list(self.param_ranges.keys())
        crossover_point = random.randint(1, len(keys) - 1)

        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2


    def mutate(self, chromosome, mutation_type='local'):
        if random.random() < self.mutation_rate:
            param_to_mutate = random.choice(list(self.param_ranges.keys()))
            param_range = self.param_ranges[param_to_mutate]
            current_val = chromosome[param_to_mutate]

            if mutation_type == 'local':
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            elif mutation_type == 'global':
                new_val = random.uniform(*param_range)
            else: # Fallback to local if unknown type
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            
            chromosome[param_to_mutate] = self._constrain_param(param_to_mutate, new_val)
        return chromosome


    def evolve(self, current_generation_delta_amps):
        if len(current_generation_delta_amps) != len(self.population):
            raise ValueError("O número de resultados de delta_amp não corresponde ao tamanho da população.")

        current_generation_best_fitness_local = -float('inf') # <--- NOVO: Melhor fitness desta geração
        
        for i, individual in enumerate(self.population):
            individual_fitness = self.calculate_fitness(current_generation_delta_amps[i])
            individual['fitness'] = individual_fitness # Atribui o fitness ao indivíduo
            
            # Atualiza o melhor fitness desta GERAÇÃO
            if individual_fitness > current_generation_best_fitness_local:
                current_generation_best_fitness_local = individual_fitness

            # Atualiza o MELHOR FITNESS GLOBAL (acumulado ao longo de todas as gerações)
            if individual_fitness > self.best_fitness:
                self.best_fitness = individual_fitness
                # Ao atualizar best_individual, copie apenas os parâmetros, não o fitness temporário
                self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                # O fitness do best_individual armazenado também deve ser o best_fitness global
                self.best_individual['fitness'] = self.best_fitness


        # <--- NOVO: Adiciona o melhor fitness desta geração ao histórico
        # Isto é importante para a checagem de convergência no main.py,
        # que compara o melhor fitness da geração atual (que é o last item no history)
        # com o melhor da geração anterior.
        self.fitness_history.append(current_generation_best_fitness_local) 

        new_population = []
        
        # Elite (o melhor indivíduo global) é mantido
        if self.best_individual and self.best_fitness > -float('inf'):
            elite_chromosome = {k: self.best_individual[k] for k in self.param_ranges.keys()}
            new_population.append(elite_chromosome)

        # Preencher o restante da nova população mantendo a proporção de mutação
        num_to_generate = self.population_size - len(new_population)
        
        # Define a proporção de mutação local/global na nova população
        # Se você quiser 50% de cada tipo de mutação:
        num_local_mutations = num_to_generate // 2
        num_global_mutations = num_to_generate - num_local_mutations # Garante que o total seja num_to_generate

        # Gera filhos com mutação "local" (próximo ao ponto atual)
        for _ in range(num_local_mutations):
            parent1, parent2 = self.select_parents()
            child = random.choice(self.crossover(parent1, parent2)) # Pega um dos filhos aleatoriamente
            child = self.mutate(child, mutation_type='local')
            new_population.append(child)

        # Gera filhos com mutação "global" (exploração ampla)
        for _ in range(num_global_mutations):
            parent1, parent2 = self.select_parents()
            child = random.choice(self.crossover(parent1, parent2)) # Pega um dos filhos aleatoriamente
            child = self.mutate(child, mutation_type='global')
            new_population.append(child)

        random.shuffle(new_population) # Mistura para evitar vieses de ordem
        self.population = new_population[:self.population_size] # Garante o tamanho correto da população

        # Retorna a população para a próxima simulação, sem o campo 'fitness'
        return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]