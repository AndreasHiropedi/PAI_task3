"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, DotProduct, RBF
from scipy.stats import norm


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # Kernel for f: Matern with recommended parameters
        kernel_f = ConstantKernel(1.0) * Matern(length_scale=10.0, nu=2.5) +WhiteKernel(noise_level=0.15) + RBF(length_scale=1.0)

        # Kernel for v: Combination of Linear + Matern
        kernel_v = (ConstantKernel(1.0) * DotProduct() + Matern(length_scale=10.0, nu=2.5) + WhiteKernel(noise_level=0.0001)) + RBF(length_scale=1.0)
        
        # Initialize Gaussian Process models
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f, alpha=0.015**2)  # Observational noise for f
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v, alpha=0.0001**2)  # Observational noise for v
        
        # Observations
        self.observations_x = []
        self.observations_f = []
        self.observations_v = []

        # Iteration tracking
        self.iteration = 0
        self.total_iterations = 100

        # Penalty for constraint violation
        self.lambda_penalty = 10.0

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # Find the point that maximizes the acquisition function
        next_point = self.optimize_acquisition_function()
        
        # Update iteration count for adaptive kappa
        self.iteration += 1
        
        return next_point

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        # Predict mean and standard deviation for f and v
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        
        # Maximum tolerated SA value
        constraint_threshold = 4.0
        
        # Calculate kappa with slower linear decay
        kappa = 2.0 * max(1 - self.iteration / self.total_iterations, 0.1)
        
        # UCB Component (for minimization)
        ucb = mean_f - kappa * std_f
        
        # Lagrangian penalty for constraint violation
        constraint_violation = np.maximum(mean_v - constraint_threshold, 0)
        penalty_term = self.lambda_penalty * constraint_violation
        
        # Adjusted acquisition function with penalty
        acquisition_value = ucb - penalty_term
        
        # Probability of satisfying the constraint (for further safe exploration)
        prob_constraint = norm.cdf((constraint_threshold - mean_v) / std_v)
        
        # Combine acquisition with constraint probability
        return (acquisition_value * prob_constraint).flatten()

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # Append new observations
        self.observations_x.append(x)
        self.observations_f.append(f)
        self.observations_v.append(v)
        
        # Update Gaussian Processes with new observations
        X = np.array(self.observations_x).reshape(-1, 1)
        F = np.array(self.observations_f).reshape(-1, 1)
        V = np.array(self.observations_v).reshape(-1, 1)
        
        self.gp_f.fit(X, F)
        self.gp_v.fit(X, V)

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """

        if not self.observations_x:
        # Return a default value within the domain if there are no observations
            return 0.0

        max_f = -np.inf
        optimal_x = None
        for x, f, v in zip(self.observations_x, self.observations_f, self.observations_v):
            if v <= 0 and f > max_f:  # Assuming v <= 0 is the constraint
                max_f = f
                optimal_x = x

        # Return optimal_x if found, otherwise a default value
        return optimal_x if optimal_x is not None else 0.0

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
