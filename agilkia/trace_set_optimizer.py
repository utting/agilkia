"""
Optimizer to choose a subset of a set of traces, based on some built-in metrics (objective functions) or custom metrics
(objective functions) provided.

Author: Shane Feng, 2021

"""
import math
import random
import numpy as np
import numpy
from typing import List, Union, Callable, Tuple, Optional
from agilkia.json_traces import TraceSet


class ObjectiveFunction:
    """
    An abstract class for objective functions that can evaluate a solution based on the chosen metrics.

    If a user aims to implement a custom objective function and use it to evaluate the solution, create a subclass of
    this abstract class. See more instruction in the documentation of the constructor and evaluate method below.
    """

    # TODO: Empty init. Create set_traces method to store data. Same for optimizer
    def __init__(self, trace_set: TraceSet, num_of_traces: int):
        """
        Constructor for an objective function.

        Use the constructor to pre compute any meta data objective function needs to evaluate a solution and store it.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            num_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        self.trace_set = trace_set
        self.num_of_traces = num_of_traces

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution and return an objective value. Implement in subclass

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns: The objective value of the solution
        """
        return 0


class FrequencyCoverage(ObjectiveFunction):
    """
    An objective function that calculates the frequency coverage of the traces selected in the solution, out of the
    total trace set frequency.

    Typically, if the traces in the trace set are generated using agilkia.SmartSequenceGenerator, there would be a
    frequency meta data associated with it. The coverage of the frequency of the selected traces, out of the total
    traces in the trace set can be used as a metric to measure the goodness of a solution.
    If the traces are not generated with SmartSequenceGenerator or there is no frequency information, please use
    other objective functions or create your own custom function.
    """

    def __init__(self, trace_set: TraceSet, num_of_traces: int):
        """
        Constructor for a frequency coverage objective function.

        Pre calculates the frequency coverage for each trace in the trace set and store it in a list.
        Pre calculates the total frequency coverage.
        If there is no frequency information, throw an exception.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            num_of_traces (int): The number of traces wanted to be selected from the trace set.
        """
        super().__init__(trace_set, num_of_traces)
        self.frequencies = np.array([trace.get_meta("freq", 0) for trace in trace_set])

        try:
            self.total_frequency_coverage = sum(self.frequencies)
            if self.total_frequency_coverage == 0:
                raise ValueError("There is no frequency information of the traces")
        except ValueError as error:
            print(error)
            raise error

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the frequency coverage of the traces selected in the solution, out of the
        total trace set frequency.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the frequency coverage of the selected traces out of the total frequency of the trace set
        """
        # If the number of selected traces in a solution is more than the number of traces wanted, return negative value
        if np.sum(solution) > self.num_of_traces:
            return (np.sum(solution) - self.num_of_traces) * -1
        solution_frequency_coverage = sum(np.array(self.frequencies) * solution)
        return solution_frequency_coverage / self.total_frequency_coverage


class EventCoverage(ObjectiveFunction):
    """
    An objective function that calculates the coverage of the individual events of the selected traces in the
    solution, out of all the traces in the trace set. Typical usage would be action coverage, status coverage and action
    status coverage. If an Event does not have a status in the output, status would be 0.
    """

    def __init__(self, trace_set: TraceSet, num_of_traces: int, event_to_str: Optional[Callable] = None):
        """
        Constructor for the event coverage objective function.

        Pass in a lambda function to extract the data on the events in the trace and use it to precalculate coverage of
        each trace in the trace set, and the total coverage of the trace set.

        To calculate action coverage, pass in event_to_string=lambda ev: ev.action.
        To calculate status coverage, pass in event_to_string=lambda ev: str(ev.status).
        To calculate action_status coverage, pass in event_to_string = lambda ev: ev.action + "_" + str(ev.status).

        Args:
            trace_set (TraceSet): The set of traces.
            num_of_traces (int): The number of traces to be selected as a test suite.
            event_to_str (Callable): The function used to extract data on the events in the trace.
        """
        super().__init__(trace_set, num_of_traces)
        self.event_to_str = event_to_str
        if self.event_to_str is None:
            self.event_to_str = (lambda ev: ev.action)
        self.trace_coverage = []
        self.total_coverage = set()
        for trace in self.trace_set:
            trace_coverage = set(trace.action_counts(event_to_str=self.event_to_str).keys())
            self.trace_coverage.append(trace_coverage)
            self.total_coverage = self.total_coverage.union(trace_coverage)
        self.trace_coverage = np.array(self.trace_coverage)

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the selected coverage of the selected traces in the solution,
        out of all the traces in the trace set.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the selected coverage of the selected traces out of the total action status
                coverage of the trace set
        """
        if np.sum(solution) > self.num_of_traces:
            return (np.sum(solution) - self.num_of_traces) * -1
        solution_action_status_coverage = set()
        solution = np.array(solution, dtype=bool)
        for trace_coverage in self.trace_coverage[solution]:
            solution_action_status_coverage = solution_action_status_coverage.union(trace_coverage)
        return len(solution_action_status_coverage) / len(self.total_coverage)


class EventPairCoverage(ObjectiveFunction):
    """
    An objective function that calculates the coverage of the event pairs of the selected traces in the solution,
    out of all the traces in the trace set. Typical usage would be action pair coverage, status pair coverage and
    action status pair coverage. If an Event does not have a status in the output, status would be 0.
    For example, for a trace, the events in a trace are "unlock, scan, scan, checkout". The action pairs would be
    "unlock_scan, scan_scan, scan_checkout".
    """

    def __init__(self, trace_set: TraceSet, num_of_traces: int, event_to_str: Optional[Callable] = None):
        """
        Constructor for the event pair coverage objective function.

        Pass in a lambda function to extract the data on the events in the trace and use it to precalculate coverage of
        each trace in the trace set, and the total coverage of the trace set.

        To calculate action pair coverage, pass in event_to_string=lambda ev: ev.action.
        To calculate status pair coverage, pass in event_to_string=lambda ev: str(ev.status).
        To calculate action_status pair coverage, pass in event_to_string = lambda ev: ev.action + "_" + str(ev.status).

        Args:
            trace_set (TraceSet): The set of traces.
            num_of_traces (int): The number of traces to be selected as a test suite.
            event_to_str (Callable): The function used to extract data on the events in the trace.
        """
        super().__init__(trace_set, num_of_traces)
        self.event_to_str = event_to_str
        if self.event_to_str is None:
            self.event_to_str = (lambda ev: ev.action)
        self.trace_coverage = []
        self.total_coverage = set()
        for trace in self.trace_set:
            trace_coverage = set()
            for i in range(len(trace) - 1):
                trace_coverage.add(event_to_str(trace[i]) + "_" + event_to_str(trace[i + 1]))
            self.trace_coverage.append(trace_coverage)
            self.total_coverage = self.total_coverage.union(trace_coverage)
        self.trace_coverage = np.array(self.trace_coverage)

    def evaluate(self, solution: numpy.ndarray) -> float:
        """
        Evaluate the solution based on the selected coverage of the selected traces in the solution,
        out of all the traces in the trace set.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            The percentage of the selected coverage of the selected traces out of the total action status
                coverage of the trace set
        """
        if np.sum(solution) > self.num_of_traces:
            return (np.sum(solution) - self.num_of_traces) * -1
        solution_action_pair_coverage = set()
        solution = np.array(solution, dtype=bool)
        for trace_coverage in self.trace_coverage[solution]:
            solution_action_pair_coverage = solution_action_pair_coverage.union(trace_coverage)
        return len(solution_action_pair_coverage) / len(self.total_coverage)


class TraceSetOptimizer:
    """
    An abstract class for different optimizers with different optimization algorithms for a given trace set.

    Given a trace set, an (or a list of) objective function(s), and the number of traces to be extracted,
    the optimizer runs the optimization algorithms and tries to extract a set of traces that maximize the objective
    values returned by the objective functions, scaled by the number of objective functions (equal weights for each).
    """

    def __init__(self, trace_set: TraceSet,
                 objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 number_of_traces: int):
        """
        Constructor for a trace set optimizer.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            objective_functions (ObjectiveFunction or List[ObjectiveFunction]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                Use ObjectiveFunction class to create your custom objective function. See more in the documentation of
                that class.

            number_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        # TODO: Rename number of traces. It's ambiguous
        self.trace_set = trace_set
        self.objective_functions = [objective_functions] if not isinstance(objective_functions,
                                                                           list) else objective_functions
        self.number_of_traces = number_of_traces
        try:
            for objective_function in self.objective_functions:
                if not isinstance(objective_function, ObjectiveFunction):
                    raise ValueError(
                        "Please provide valid built in objective functions or your custom objective functions")
        except ValueError as error:
            print(error)
            raise error

    def objective(self, solution: np.ndarray) -> float:
        """
        Evaluate the solution with all objective functions passed in, assign an equal weight to them based on the
        number of objective functions passed in and combine them.

        Args:
            solution (numpy.ndarray): A binary vector with the same length of the trace set that represent the solution.
                On every position, 1 means the trace at the position is selected, and 0 means not selected.

        Returns:
            Combined the different objective values with an equal weight based on the number of the objective
                functions, and return it. Result would be between 0 and 1
        """
        number_of_objective_functions = len(self.objective_functions)
        total_objective_value = 0
        for objective_function in self.objective_functions:
            objective_value = objective_function.evaluate(solution)
            total_objective_value += objective_value * (1 / number_of_objective_functions)
        return total_objective_value


class GreedyOptimizer(TraceSetOptimizer):
    """
    A subclass of TraceSetOptimizer that uses the Greedy Search Algorithm to search for a subset of traces that tries to
    maximize the objective value
    """

    def __init__(self, trace_set: TraceSet,
                 objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 number_of_traces: int):
        """Creates an optimizer that uses the Greedy Search Algorithm to search for a subset of traces that tries to
        maximize the objective value

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            objective_functions (ObjectiveFunction or List[ObjectiveFunction]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                Use ObjectiveFunction class to create your custom objective function. See more in the documentation of
                that class.
            number_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
        """
        super().__init__(trace_set=trace_set, objective_functions=objective_functions,
                         number_of_traces=number_of_traces)

    def optimize(self) -> Tuple[TraceSet, float]:
        """ Implements the greedy search algorithm and applies it on the trace set passed in.
        It loops through each of the not selected trace so far, adds it to the current solution, evaluates the
        solution and records the best objective value achieved by a solution. After one iteration, it selects the
        trace that results in a solution that achieves the best objective value, and go on to the next iteration.
        The algorithm stops when it reaches the number of traces wanted.

        Returns:
            The algorithm returns the best trace set it found and the objective value the solution achieves.
        """
        print("Starting Greedy Search...Selecting", self.number_of_traces, "traces")
        num_of_variables = len(self.trace_set)
        solution = np.zeros(num_of_variables)
        best_objective_value = 0
        best_index = None

        for j in range(self.number_of_traces):
            for i in range(num_of_variables):
                if solution[i] != 1:
                    solution[i] = 1
                    objective_value = self.objective(solution)
                    if objective_value > best_objective_value:
                        best_objective_value = objective_value
                        best_index = i
                    solution[i] = 0
            solution[best_index] = 1
        selected_traces = [self.trace_set[i] for i in range(num_of_variables) if solution[i]]
        selected_traces = TraceSet(selected_traces)
        return selected_traces, best_objective_value


class ParticleSwarmOptimizer(TraceSetOptimizer):
    """
    A subclass of TraceSetOptimizer that uses the Particle Swarm Optimization Algorithm to search for a subset of
    traces that tries to maximize the objective value
    """

    def __init__(self, trace_set: TraceSet,
                 objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 number_of_traces: int, number_of_particles: int, number_of_iterations: int, c1: float, c2: float):
        """Creates an optimizer that uses the Particle Swarm Optimization Algorithm to search for a subset of traces
        that tries to maximize the objective value. This is using the Binary version of PSO.

        Args:
            trace_set (TraceSet): A set of traces chosen or output by some kind of model.
            objective_functions (Callable or List[Callable]): The objective functions to be used to
                evaluate the chosen subset of traces. Built in functions or custom defined functions are acceptable.
                An objective function would take in three arguments, which are the trace set, a binary vector to
                represent the solution (on every position, 1 means selected), and the number of traces wanted to be
                selected. Any external variables needed for the objective function should be global variables that can
                be accessed within the function. This is for the convenience to compute different objective values with
                different objective functions and combine them.
            number_of_traces (int): The number of traces wanted to be selected from the trace set.
                For example, if 10 out of 20 traces in the trace set are wanted to be returned as a test suite, the
                optimizer runs the algorithms and tries to choose 10 traces from the trace set that maximize the
                objective values returned by the objective functions.
            number_of_particles (int): Number of particles used in the algorithm
            number_of_iterations (int): Number of iterations of the algorithm
            c1 (float): Controlling parameter of the influence of the particle's personal best position on the
                particle's velocity during update
            c2 (float): Controlling parameter of the influence of the population's global best position on the
                particle's velocity during update
        """
        super().__init__(trace_set=trace_set, objective_functions=objective_functions,
                         number_of_traces=number_of_traces)
        self.number_of_iterations = number_of_iterations
        self.number_of_particles = number_of_particles
        self.c1 = c1
        self.c2 = c2

    def optimize(self) -> Tuple[TraceSet, float]:
        """ Implements the Particle Swarm Optimization Algorithm and applies it on the trace set passed in.
        A particle's position is a solution that the particle holds. The algorithm initialise the positions to be a
        binary vector that holds 0 and 1 randomly. The velocity if used to update the position of a particle. A particle
        also holds a personal best position and a personal best value that the particle has been to and achieved. The
        population holds a global best position and a global best value.
        During iterations, the particle compares it's current objective value with the personal best value, and
        substitute the global best position with the current position if it's current value is higher. It would also do
        the same to the global best position and global best value.
        During velocity update, w, c1, c2 are controlling parameters to control the influence of current velocity,
        personal best and global best on the update. After update, if the velocity exceeds to maximum or minimum,
        set it to maximum mor minimum velocity.
        After updating velocity, transform the velocity vector into sigmoid values. Generate a random vector between 0
        and 1. If the sigmoid values are larger than the generated values at an index position, change it to 1. This
        final vector will be the particle's new position.

        Returns:
            The algorithm returns the best trace set it found and the objective value the solution achieves.
        """
        print("Starting Particle Swarm with", self.number_of_particles, "particles,", self.number_of_iterations,
              "iterations,", "c1 as", self.c1, ", c2 as", self.c2, ", selecting", self.number_of_traces, "traces")
        # TODO: Should these all be put in self?
        num_of_variables = len(self.trace_set)

        # Every bit in the solution can only be either 0 or 1
        upper_bound = np.ones(num_of_variables)
        lower_bound = np.zeros(num_of_variables)

        # Define the upper bound and lower bound of the controlling parameter of the influence for the previous
        # velocity on the particle's velocity during update
        w_max = 0.9
        w_min = 0.1

        # Define maximum and minimum velocity to avoid exceeding this limit. The value 6 is suggested by the author of
        # the algorithm
        v_max = np.ones(num_of_variables) * 6
        v_min = -v_max

        # Initialise the particle container, global best position, and global best objective value
        particles = []
        gbest_x = np.zeros(num_of_variables)
        gbest_val = -math.inf

        # Initialise the particle population
        for i in range(self.number_of_particles):
            particle = {}
            # Round positions to nearest integers
            particle['X'] = np.rint(
                np.add(np.subtract(upper_bound, lower_bound) * np.random.rand(num_of_variables), lower_bound))
            # Velocity
            particle['V'] = np.zeros(num_of_variables)
            # Position of a particle, this is the solution the particle holds
            particle['PBESTX'] = np.zeros(num_of_variables)
            # Objective value of the solution
            particle['PBESTO'] = -math.inf
            particles.append(particle)

        # Start iteration
        for t in range(self.number_of_iterations):
            if t % 100 == 0:
                print(t, "iterations. Current global best:", gbest_val)

            # Update personal best and global best
            for index, particle in enumerate(particles):
                current_x = particle['X']
                particle["O"] = self.objective(current_x)

                if particle['O'] > particle['PBESTO']:
                    particle['PBESTX'] = current_x
                    particle['PBESTO'] = particle["O"]

                if particle['O'] > gbest_val:
                    gbest_x = current_x
                    gbest_val = particle['O']

            # Update particle's position and velocity
            # w would change adaptively according the current iteration
            w = w_max - t * ((w_max - w_min) / self.number_of_iterations)

            for index, particle in enumerate(particles):
                particle['V'] = w * particle['V'] + \
                                self.c1 * np.random.rand(num_of_variables) * np.subtract(particle['PBESTX'],
                                                                                         particle['X']) + \
                                self.c2 * np.random.rand(num_of_variables) * np.subtract(gbest_x, particle['X'])

                # Check min max velocity
                index_greater = np.where(particle['V'] > v_max)[0].tolist()
                index_smaller = np.where(particle['V'] < v_min)[0].tolist()
                if index_greater:
                    for idx in index_greater:
                        particle['V'][idx] = v_max[idx]
                if index_smaller:
                    for idx in index_smaller:
                        particle['V'][idx] = v_min[idx]

                # Sigmoid transfer
                sigmoid = 1 / (1 + np.exp(-particle['V']))
                temp = np.random.rand(num_of_variables) < sigmoid
                temp = temp * 1
                particle['X'] = temp
        selected_traces = [self.trace_set[i] for i in range(num_of_variables) if gbest_x[i]]
        selected_traces = TraceSet(selected_traces)
        return selected_traces, gbest_val


class GeneticOptimizer(TraceSetOptimizer):
    def __init__(self, trace_set: TraceSet,
                 objective_functions: Union[ObjectiveFunction, List[ObjectiveFunction]],
                 number_of_traces: int, number_of_iterations: int, number_of_chromosomes: int, prob_cross: float,
                 prob_mutate: float, elitism: Optional[bool] = False, elitism_rate: Optional[float] = 0):
        super().__init__(trace_set, objective_functions, number_of_traces)
        self.number_of_iterations = number_of_iterations
        self.number_of_chromosomes = number_of_chromosomes
        self.number_of_genes = len(self.trace_set)
        self.prob_cross = prob_cross
        self.prob_mutate = prob_mutate
        self.elitism = elitism
        self.elitism_rate = elitism_rate
        # Initialise population
        self.population = np.rint(np.random.rand(self.number_of_chromosomes, self.number_of_genes))
        self.new_population = np.zeros((self.number_of_chromosomes, self.number_of_genes))

    def normalise_objective_values(self):
        objective_values = np.apply_along_axis(self.objective, 1, self.population)
        min_objective_value = min(objective_values)
        if min_objective_value < 0:
            objective_values = objective_values - min_objective_value + 1
        normalised_objective_values = objective_values / np.sum(objective_values)
        return normalised_objective_values

    def roulette_wheel(self, normalised_objective_values):
        cum_sum = np.cumsum(normalised_objective_values)
        r = random.random()
        for index, condition in enumerate(r <= cum_sum):
            if condition:
                return index

    def select(self, population, normalised_objective_values, number_of_parents):
        selected_parents_indexes = []
        selected_parents = []
        for i in range(number_of_parents):
            index = self.roulette_wheel(normalised_objective_values)
            while index in selected_parents_indexes:
                index = self.roulette_wheel(normalised_objective_values)
            selected_parents_indexes.append(index)
            selected_parents.append(population[index])
        selected_parents = np.array(selected_parents)
        return selected_parents

    def crossover(self, parent1, parent2, crossover_name):
        child1 = None
        child2 = None
        if crossover_name == "single":
            crossover_point = random.randint(1, self.number_of_genes - 2)
            child1 = np.concatenate([parent1[0:crossover_point], parent2[crossover_point: self.number_of_genes]])
            child2 = np.concatenate([parent2[0:crossover_point], parent1[crossover_point: self.number_of_genes]])

        if crossover_name == "double":
            crossover_point1 = random.randint(1, self.number_of_genes - 2)
            crossover_point2 = random.randint(1, self.number_of_genes - 2)
            while crossover_point1 == crossover_point2:
                crossover_point2 = random.randint(1, self.number_of_genes - 2)
            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
            child1 = np.concatenate([parent1[0:crossover_point1], parent2[crossover_point1: crossover_point2],
                                     parent1[crossover_point2:self.number_of_genes]])
            child2 = np.concatenate([parent2[0:crossover_point1], parent1[crossover_point1: crossover_point2],
                                     parent2[crossover_point2:self.number_of_genes]])

        r1 = random.random()
        child1 = child1 if r1 <= self.prob_cross else parent1
        r2 = random.random()
        child2 = child2 if r2 <= self.prob_cross else parent2
        return child1, child2

    def mutate(self, child):
        for i in range(self.number_of_genes):
            r = random.random()
            if r <= self.prob_mutate:
                child[i] = not child[i]
        return child

    def elitism(self):
        number_of_elites = int(self.number_of_chromosomes * self.elitism_rate)
        objective_values = np.apply_along_axis(self.objective, 1, self.population)
        new_pop_objective_values = np.apply_along_axis(self.objective, 1, self.new_population)

        for i in range(number_of_elites):
            max_index = np.argmax(objective_values)
            elite = self.population[max_index]
            min_index = np.argmin(new_pop_objective_values)
            if min(new_pop_objective_values) < max(objective_values):
                self.new_population[min_index] = elite
                new_pop_objective_values = np.delete(new_pop_objective_values, min_index, 0)
                self.population = np.delete(self.population, max_index, 0)
                objective_values = np.delete(objective_values, max_index, 0)
            else:
                break
        return self.new_population

    def optimize(self):
        for i in range(self.number_of_iterations):
            normalised_objective_values = self.normalise_objective_values()

        for i in range(0, self.number_of_chromosomes, 2):
            # Selection
            [parent1, parent2] = self.select(self.population, normalised_objective_values, 2)
            # Crossover
            child1, child2 = crossover(parent1, parent2, prob_cross, "double")
            # Mutation
            child1 = mutate(child1, prob_mutate)
            child2 = mutate(child2, prob_mutate)
            newPopulation[i] = child1
            newPopulation[i + 1] = child2