import numpy as np
import random

class GreyWolfOptimizer:
    def __init__(self, dist_matrix, num_wolves=100, max_iter=100):
        self.dist_matrix = np.array(dist_matrix)
        self.num_cities = len(dist_matrix)
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.wolves = np.array([self._generate_random_route() for _ in range(self.num_wolves)])
        self.best_route = None
        self.best_length = float('inf')

    def _generate_random_route(self):
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route

    def _calculate_total_distance(self, route):
        """Calculate the total distance for the route."""
        distance = 0
        for i in range(len(route) - 1):
            distance += self.dist_matrix[route[i], route[i + 1]]
        distance += self.dist_matrix[route[-1], route[0]]  # Return to the starting city
        return distance

    def _fitness(self, route):
        return self._calculate_total_distance(route)

    def _update_position(self, wolf, alpha, beta, delta, a):
        """Update the position of the current wolf based on alpha, beta, and delta wolves."""
        new_wolf = wolf.copy()
        
        for i in range(self.num_cities):
            if random.random() < 0.5:
                # Exploration: random swap
                swap_idx1, swap_idx2 = random.sample(range(self.num_cities), 2)
                new_wolf[swap_idx1], new_wolf[swap_idx2] = new_wolf[swap_idx2], new_wolf[swap_idx1]
            else:
                # Exploitation: refine using alpha, beta, delta
                chosen_wolf = random.choice([alpha, beta, delta])
                swap_idx1 = random.randint(0, self.num_cities - 1)
                new_wolf[swap_idx1] = chosen_wolf[swap_idx1]
        
        return new_wolf

    def solve(self):
        best_route, best_length = None, float('inf')

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Linearly decreasing factor

            # Evaluate fitness for all wolves
            fitness = np.array([self._fitness(wolf) for wolf in self.wolves])

            # Find the alpha, beta, and delta wolves
            sorted_indices = np.argsort(fitness)
            alpha_idx, beta_idx, delta_idx = sorted_indices[:3]

            # Update best solution
            if fitness[alpha_idx] < best_length:
                best_length = fitness[alpha_idx]
                best_route = self.wolves[alpha_idx]

            # Update wolves' positions
            new_wolves = []
            for i in range(self.num_wolves):
                wolf = self.wolves[i]
                alpha = self.wolves[alpha_idx]
                beta = self.wolves[beta_idx]
                delta = self.wolves[delta_idx]

                new_wolf = self._update_position(wolf, alpha, beta, delta, a)
                new_wolves.append(new_wolf)

            self.wolves = np.array(new_wolves)

            print(f"Iteration {t + 1}: Best Length = {best_length:.2f}")

        return best_route, best_length

# Example usage
if __name__ == "__main__":
    # Example distance matrix (symmetric matrix for TSP)
    dist_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    gwo = GreyWolfOptimizer(dist_matrix, num_wolves=10, max_iter=50)
    best_route, best_length = gwo.solve()

    print("Best Route:", best_route)
    print("Best Length:", best_length)
