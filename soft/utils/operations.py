import copy
import random
import matplotlib.pyplot as plt

from soft.utils.individual import Individual


def selection_rank_with_elite(individuals, elite_size = 0):
    sorted_individuals = sorted(individuals, key = lambda ind: ind.fitness, reverse = True)
    rank_distance = 1 / len(individuals)
    ranks = [(1 - i * rank_distance) for i in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[0:elite_size]

    for i in range(len(sorted_individuals) - elite_size):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break

    return selected


def crossover(ind1, ind2):
    child1 = Individual(copy.deepcopy(ind1.gene_tree), ind1.structure)
    child2 = Individual(copy.deepcopy(ind2.gene_tree), ind2.structure)
    rand1 = child1.get_random_node()
    rand2 = child2.get_random_node_by_operator_type(rand1.name.o_type)
    if not rand2:
        return [child1, child2]
    child1.replace_node(rand1, rand2)
    child2.replace_node(rand2, rand1)
    child1.set_up()
    child2.set_up()
    return [child1, child2]


def mutate(ind, operator_type_ratio, min_h, max_h):
    ind_copy = Individual(copy.deepcopy(ind.gene_tree), ind.structure)
    random_types = random.choices(list(operator_type_ratio.keys()), list(operator_type_ratio.values()))
    random_type = random_types[0]

    if random_type != 'top':
        replaced_node = ind_copy.get_random_node_by_operator_type(random_type)

        if not replaced_node:
            return ind_copy
        operator = replaced_node.name
        domain = operator.return_type
        if operator.is_term():
            operator.value = operator.mutation_rule(operator.value)
        else:
            random_tree = ind.structure.generate_random_tree(domain, min_h, max_h)
            ind_copy.replace_node(replaced_node, random_tree)
    else:
        random_tree = ind.structure.generate_random_tree('bool', min_h, max_h, starts_with = 'OR',
                                                         put_before_top = ind_copy.gene_tree.name)
        ind_copy = Individual(copy.deepcopy(random_tree), ind.structure)

    ind_copy.set_up()
    return ind_copy


def mutate_operator(ind):
    ind_copy = Individual(copy.deepcopy(ind.gene_tree), ind.structure)
    random_types = ['operator']
    replaced_node = ind_copy.get_random_node_by_operator_type(random_types[0])
    operator = replaced_node.name
    operator.value = operator.mutation_rule(operator.value)
    ind_copy.set_up()
    return ind_copy


def crossover_fitness_driven(p1, p2):
    c1, c2 = crossover(p1, p2)
    candidates = [c1, c2, p1, p2]
    best = sorted(candidates, key = lambda ind: ind.fitness, reverse = True)
    return best[0:2]


def mutation_fitness_driven(ind, operator_kind_ratio, min_h, max_h, max_tries = 3):
    for _ in range(0, max_tries):
        mutated = mutate(ind, operator_kind_ratio, min_h, max_h)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind


def mutation_operator_fitness_driven(ind, max_tries = 3):
    for _ in range(0, max_tries):
        mutated = mutate_operator(ind)
        if mutated.fitness > ind.fitness:
            return mutated
    return ind


def crossover_operation(population, prob):
    crossed_offspring = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < prob:
            kid1, kid2 = crossover_fitness_driven(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)
    return crossed_offspring


def mutation_operation(population, operator_kind_ratio, min_h, max_h, prob, with_operator_mutation = False):
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = mutation_fitness_driven(mutant, operator_kind_ratio, min_h, max_h)
            mutated_offspring.append(new_mutant)
        else:
            if with_operator_mutation:
                new_mutant = mutation_operator_fitness_driven(mutant, max_tries = 10)
                mutated_offspring.append(new_mutant)
            else:
                mutated_offspring.append(mutant)
    return mutated_offspring


def stats(population, best_ind, fit_avg, fit_best):
    best_of_generation = max(population, key = lambda ind: ind.fitness)
    if best_ind.fitness < best_of_generation.fitness:
        best_ind = best_of_generation
    fit_avg.append(sum([ind.fitness for ind in population]) / len(population))
    fit_best.append(best_ind.fitness)

    return best_ind, fit_avg, fit_best


def plot_stats(fit_avg, fit_best, title):
    plt.plot(fit_avg, label = "Average Fitness of Generation")
    plt.plot(fit_best, label = "Best Fitness")
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.show()
    plt.close()
