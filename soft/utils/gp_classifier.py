import random

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from soft.utils.individual import Individual
from soft.utils.operations import selection_rank_with_elite, crossover_operation, mutation_operation, stats


class GpClassifier(BaseEstimator, ClassifierMixin):
    cache = {}

    def __init__(self,
                 space,
                 population_size = 50,
                 max_generations = 50,
                 no_improvement_stop = 10,
                 number_of_populations = 4,
                 population_exchange = 5,
                 with_operator_mutation = False
                 ):
        self.space = space
        self.population_size = population_size
        self.max_generations = max_generations
        self.no_improvement_stop = no_improvement_stop
        self.number_of_populations = number_of_populations
        self.population_exchange = population_exchange
        self.with_operator_mutation = with_operator_mutation

    def fit(self, X, y):

        self.__class__.cache = {}

        num_rows, num_cols = X.shape

        def fitness_function(ind):

            if ind.signature in self.__class__.cache:
                return self.__class__.cache[ind.signature]

            if ind.count_genes() > num_cols * 16:
                return 0

            param_map = {}
            for i in range(num_cols):
                param_map[f'P{i+1}_'] = X[:, i]

            predicted_pure = ind.evaluate(**param_map)

            try:
                x_len = len(predicted_pure)
            except:
                x_len = 1

            if x_len == 1:
                predicted = [int(round(min(max(predicted_pure, 0), 1)))] * len(X)
            else:
                predicted = [int(round(min(max(v, 0), 1))) for v in predicted_pure]

            real = y.astype(int)
            fitness = balanced_accuracy_score(real, predicted)
            self.__class__.cache[ind.signature] = fitness
            return fitness

        Individual.set_fitness_function(fitness_function)

        def create_random(space, return_type, min_height, max_height, starts_with = None):
            return Individual(
                space.generate_random_tree(return_type, min_height, max_height, starts_with = starts_with),
                space
            )

        space = self.space
        POPULATION_SIZE = self.population_size
        CROSSOVER_PROBABILITY = .5
        MUTATION_PROBABILITY = .5
        MAX_GENERATIONS = self.max_generations

        min_height = {'logic': 1, 'compare': 1, 'func': 1}
        max_height = {'logic': round(num_cols / 4), 'compare': 1, 'func': 2}

        populations = [None] * self.number_of_populations
        best_inds = [None] * self.number_of_populations

        for p_i in range(self.number_of_populations):
            first_population = [
                create_random(space, 'bool', min_height, max_height, starts_with = 'OR')
                for _ in range(POPULATION_SIZE)
            ]
            best_inds[p_i] = random.choice(first_population)
            populations[p_i] = first_population.copy()

        fit_avg = []
        fit_best = []
        generation_num = 0
        last_fitness = 0
        no_improvement = 0
        while generation_num < MAX_GENERATIONS and no_improvement < self.no_improvement_stop and last_fitness < 1:
            generation_num += 1
            print('===')
            print(f'Generation {generation_num}')

            for p_i in range(self.number_of_populations):
                offspring = selection_rank_with_elite(populations[p_i], elite_size = 1)
                crossed_offspring = crossover_operation(offspring, CROSSOVER_PROBABILITY)
                mutated_offspring = mutation_operation(
                    crossed_offspring,
                    {'top': 10, 'logic': 3, 'compare': 5, 'func': 10, 'real': 50},
                    {'logic': 1, 'compare': 1, 'func': 1},
                    {'logic': round(num_cols / 4), 'compare': 1, 'func': 3},
                    MUTATION_PROBABILITY,
                    with_operator_mutation = self.with_operator_mutation
                )
                populations[p_i] = mutated_offspring.copy()
                best_ind, fit_avg, fit_best = stats(populations[p_i], best_inds[p_i], fit_avg, fit_best)
                best_inds[p_i] = best_ind

                print('=')
                print(f'population {p_i}: {round(fit_best[-1], 6)}, {round(fit_avg[-1], 6)}')
                print(f'({best_ind.count_genes()}) - {best_ind.signature}')

            if generation_num % self.population_exchange == 0:
                print("exchanging the best")
                for p_i in range(self.number_of_populations):
                    populations[p_i].append(best_inds[(p_i + 1) % self.number_of_populations])

            best_ever = max(best_inds, key = lambda i: i.fitness)
            print('=====')
            print(f'BEST EVER: {round(best_ever.fitness, 6)}')

            if best_ever.fitness == last_fitness:
                no_improvement = no_improvement + 1
            else:
                no_improvement = 0
            print(f'No Improvement: {no_improvement}')

            last_fitness = best_ever.fitness

            self.best_model = best_ever

        self.__class__.cache = {}

        return self

    def predict(self, X):
        num_rows, num_cols = X.shape

        param_map = {}
        for i in range(num_cols):
            param_map[f'P{i+1}_'] = X[:, i]

        predicted_pure = self.best_model.evaluate(**param_map)

        try:
            x_len = len(predicted_pure)
        except:
            x_len = 1

        if x_len == 1:
            predicted = [int(round(min(max(predicted_pure, 0), 1)))] * len(X)
        else:
            predicted = [int(round(min(max(v, 0), 1))) for v in predicted_pure]

        return predicted

    def predict_proba(self, X):
        num_rows, num_cols = X.shape

        param_map = {}
        for i in range(num_cols):
            param_map[f'P{i+1}_'] = X[:, i]

        predicted_pure = self.best_model.evaluate(**param_map)

        try:
            x_len = len(predicted_pure)
        except:
            x_len = 1

        if x_len == 1:
            predicted = [float(min(max(predicted_pure, 0), 1))] * len(X)
        else:
            predicted = [float(min(max(v, 0), 1)) for v in predicted_pure]

        values = np.c_[predicted, predicted]

        return values
