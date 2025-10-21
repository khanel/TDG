import abc
import numpy as np
from typing import Protocol, Tuple, Iterable, Optional
from dataclasses import dataclass

@dataclass
class Elite:
    x: np.ndarray
    fitness: float
    bd: np.ndarray

class Indexer(Protocol):
    # Return a hashable key for the archive, or None if BD out of range (after clipping policy)
    def key(self, bd: np.ndarray) -> Optional[tuple]: ...
    def size(self) -> int: ...  # number of cells
    def bins(self) -> Tuple[int, ...]: ...  # grid shape or (n_centroids,)

class GridIndexer:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray, bins_per_dim: Tuple[int, ...], clip: bool = True):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        self.bins_per_dim = tuple(bins_per_dim)
        self.clip = clip
        self.dim = len(self.bins_per_dim)
        assert len(self.mins) == self.dim and len(self.maxs) == self.dim

    def key(self, bd: np.ndarray) -> Optional[tuple]:
        bd = np.asarray(bd, dtype=float)
        if self.clip:
            bd = np.clip(bd, self.mins, self.maxs)

        # Check bounds
        if np.any(bd < self.mins) or np.any(bd > self.maxs):
            return None

        # Normalize and discretize
        normalized = (bd - self.mins) / (self.maxs - self.mins)
        indices = []
        for i in range(self.dim):
            idx = int(np.floor(normalized[i] * self.bins_per_dim[i]))
            idx = max(0, min(idx, self.bins_per_dim[i] - 1))
            indices.append(idx)

        return tuple(indices)

    def size(self) -> int:
        return int(np.prod(self.bins_per_dim))

    def bins(self) -> Tuple[int, ...]:
        return self.bins_per_dim

class CVTIndexer:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray, n_cells: int, n_samples: int = 200_000, kmeans_init: str = "k-means++", rng=None):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        self.n_cells = n_cells
        self.rng = rng or np.random.default_rng()

        # Build centroids using Lloyd's algorithm approximation
        self.centroids = self._build_centroids(n_samples, kmeans_init)

    def _build_centroids(self, n_samples, kmeans_init):
        # Sample points uniformly in the BD space
        samples = self.rng.uniform(self.mins, self.maxs, (n_samples, len(self.mins)))

        # Initialize centroids
        if kmeans_init == "random":
            centroids = samples[self.rng.choice(n_samples, self.n_cells, replace=False)]
        else:  # k-means++
            centroids = self._kmeans_pp_init(samples)

        # Run Lloyd's algorithm
        for _ in range(20):  # 20 iterations
            # Assign each sample to nearest centroid
            distances = np.sum((samples[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            for k in range(self.n_cells):
                mask = assignments == k
                if np.any(mask):
                    centroids[k] = np.mean(samples[mask], axis=0)
                else:
                    centroids[k] = self.rng.uniform(self.mins, self.maxs)

        return centroids

    def _kmeans_pp_init(self, samples):
        centroids = [samples[self.rng.integers(len(samples))]]
        for _ in range(1, self.n_cells):
            distances = np.min([np.sum((samples - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = distances / np.sum(distances)
            next_centroid_idx = self.rng.choice(len(samples), p=probs)
            centroids.append(samples[next_centroid_idx])
        return np.array(centroids)

    def key(self, bd: np.ndarray) -> Optional[tuple]:
        bd = np.asarray(bd, dtype=float)
        # Clip to bounds
        bd = np.clip(bd, self.mins, self.maxs)

        # Find nearest centroid
        distances = np.sum((self.centroids - bd[None, :]) ** 2, axis=1)
        nearest_idx = int(np.argmin(distances))

        return (nearest_idx,)

    def size(self) -> int:
        return self.n_cells

    def bins(self) -> Tuple[int, ...]:
        return (self.n_cells,)

class Problem(Protocol):
    def fitness(self, x: np.ndarray) -> float: ...
    def behavior_descriptor(self, x: np.ndarray) -> np.ndarray: ...  # shape (k,)
    def sample(self) -> np.ndarray: ...  # random feasible genotype
    def mutate(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
    def crossover(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...  # optional
    # Optional latent decode for discrete reps (e.g., random-keys for permutations)
    def decode(self, z: np.ndarray) -> np.ndarray: ...  # optional

class BaseProblem(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def fitness(self, x: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def behavior_descriptor(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def sample(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mutate(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pass

    def crossover(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Default: arithmetic crossover
        alpha = rng.random()
        return alpha * a + (1 - alpha) * b

    def decode(self, z: np.ndarray) -> np.ndarray:
        # Default: identity (for continuous problems)
        return z