"""
Evolutionary Meta-Optimisation Layer (EML).

A two-timescale optimisation system that co-evolves discrete
architectural parameters alongside gradient-based weight training.

Fast timescale (every step):
    Gradient descent on continuous weights.

Slow timescale (every T_evo steps):
    Evolutionary algorithm on discrete architecture:
    - Partition structure (coherence thresholds per layer)
    - Number of attention heads per layer
    - Number of ARM-FFN experts per layer
    - CFM/AFM scaling coefficients
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import numpy as np


@dataclass
class Chromosome:
    """Encodes a complete DSTT architectural specification.

    Each gene corresponds to a per-layer architectural parameter.
    The chromosome length is n_layers * genes_per_layer.

    Genes per layer:
        [0] n_heads (int, 4..64)
        [1] n_experts (int, 2..32)
        [2] coherence_threshold (float, 0.05..0.95)
        [3] cfm_alpha (float, 0.01..1.0)
        [4] afm_beta (float, 0.01..1.0)
    """

    GENES_PER_LAYER: int = 5

    genes: np.ndarray
    """Raw gene values in [0, 1], shape (n_layers * GENES_PER_LAYER,)."""

    fitness: float = 0.0
    """Evaluated fitness score."""

    n_layers: int = 12
    """Number of model layers encoded."""

    @classmethod
    def random(cls, n_layers: int) -> Chromosome:
        """Create a random chromosome."""
        length = n_layers * cls.GENES_PER_LAYER
        return cls(
            genes=np.random.uniform(0, 1, size=length),
            n_layers=n_layers,
        )

    def decode(self) -> list[dict]:
        """Decode genes into per-layer architectural specifications.

        Returns:
            List of dicts, one per layer, with keys:
            'n_heads', 'n_experts', 'coherence_threshold',
            'cfm_alpha', 'afm_beta'.
        """
        specs = []
        g = self.GENES_PER_LAYER
        for layer in range(self.n_layers):
            offset = layer * g
            raw = self.genes[offset : offset + g]
            specs.append({
                "n_heads": int(4 + raw[0] * 60),           # [4, 64]
                "n_experts": int(2 + raw[1] * 30),          # [2, 32]
                "coherence_threshold": 0.05 + raw[2] * 0.9, # [0.05, 0.95]
                "cfm_alpha": 0.01 + raw[3] * 0.99,          # [0.01, 1.0]
                "afm_beta": 0.01 + raw[4] * 0.99,           # [0.01, 1.0]
            })
        return specs

    def copy(self) -> Chromosome:
        return Chromosome(
            genes=self.genes.copy(),
            fitness=self.fitness,
            n_layers=self.n_layers,
        )


class EvolutionaryMetaOptimiser:
    """Evolves DSTT architectural parameters via tournament
    selection, single-point crossover, and mutation.

    Usage::

        eml = EvolutionaryMetaOptimiser(config)
        eml.initialise()

        for generation in range(config.evo_generations):
            for chromo in eml.population:
                arch_spec = chromo.decode()
                # ... build model, train T_evo steps, evaluate ...
                chromo.fitness = evaluated_fitness

            eml.evolve()

        best_arch = eml.best_chromosome.decode()

    Args:
        config: Model configuration with EML hyperparameters.
    """

    def __init__(self, config):
        self.config = config
        self.population: list[Chromosome] = []
        self.generation: int = 0
        self.best_chromosome: Chromosome | None = None
        self.history: list[float] = []  # best fitness per generation

    def initialise(self) -> None:
        """Create initial random population."""
        self.population = [
            Chromosome.random(self.config.n_layers)
            for _ in range(self.config.evo_population)
        ]
        self.generation = 0

    def evolve(self) -> None:
        """Execute one generation of evolution.

        Steps:
        1. Sort population by fitness (descending).
        2. Preserve top elites.
        3. Fill remaining slots via tournament selection + crossover + mutation.
        """
        N = self.config.evo_population
        n_elite = max(1, int(self.config.evo_elitism_rate * N))
        k = self.config.evo_tournament_size
        mu = self.config.evo_mutation_rate

        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Track best
        if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = self.population[0].copy()
        self.history.append(self.population[0].fitness)

        # Elites
        new_pop = [c.copy() for c in self.population[:n_elite]]

        # Fill with offspring
        while len(new_pop) < N:
            p1 = self._tournament_select(k)
            p2 = self._tournament_select(k)
            o1, o2 = self._crossover(p1, p2)
            self._mutate(o1, mu)
            self._mutate(o2, mu)
            new_pop.append(o1)
            if len(new_pop) < N:
                new_pop.append(o2)

        self.population = new_pop[:N]
        self.generation += 1

    def _tournament_select(self, k: int) -> Chromosome:
        """Select a parent via tournament selection."""
        candidates = random.sample(self.population, min(k, len(self.population)))
        return max(candidates, key=lambda c: c.fitness)

    @staticmethod
    def _crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Single-point crossover."""
        length = len(p1.genes)
        point = random.randint(1, length - 1)
        g1 = np.concatenate([p1.genes[:point], p2.genes[point:]])
        g2 = np.concatenate([p2.genes[:point], p1.genes[point:]])
        return (
            Chromosome(genes=g1, n_layers=p1.n_layers),
            Chromosome(genes=g2, n_layers=p1.n_layers),
        )

    @staticmethod
    def _mutate(chromo: Chromosome, rate: float) -> None:
        """Per-gene mutation with uniform random replacement."""
        for i in range(len(chromo.genes)):
            if random.random() < rate:
                chromo.genes[i] = random.uniform(0, 1)

    def get_convergence_info(self) -> dict:
        """Return convergence statistics."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_chromosome.fitness if self.best_chromosome else 0.0,
            "mean_fitness": np.mean([c.fitness for c in self.population]),
            "std_fitness": np.std([c.fitness for c in self.population]),
            "history": self.history,
        }
