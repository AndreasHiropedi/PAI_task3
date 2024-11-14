"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        self.results = []

        # TODO: Define all relevant class members for your BO algorithm here.
        # Have added these are the standard deviations of the noise for f and v
        self.sd_f = 0.15
        self.sd_v = 0.0001

        # for max improvement 
        self.opt_f = 0

    def f(self, x):
        # may need to tune lengthscale
        kernel_f = Matern(nu=2.5) + WhiteKernel(0.5)
        # may need to tune alpha
        f_gp = kernel_f.__call__(x)
        return f_gp
        
    def v(self, x):
        # no idea how this gets called and where to add in sd_v
        # lengthscale can be 10, 1, or 0.5 
        kernel_v = RBF(length_scale=10) + WhiteKernel(np.sqrt(2)) + ConstantKernel(4)
        # may need to tune alpha
        # this just uses the kernel to compute covariance
        v_gp = kernel_v.__call__(x)
        return v_gp
    

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

        recommendation = self.optimize_acquisition_function()


        return recommendation

        raise NotImplementedError

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

        # using equation in task
        prop_f = self.f(x) + np.random.normal(0,self.sd_f)

        # aqcuisition function is expectation but i implemented wrong idk
        af_value = prop_f-self.opt_f 

        zeroes = np.zeros_like(x)
        return np.maximum(zeroes, af_value).item()

        # TODO: Implement the acquisition function you want to optimize.
        #raise NotImplementedError

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

        self.results.append([x,f,v])
        self.opt_f = f
        
        # TODO: Add the observed data {x, f, v} to your model.
        #raise NotImplementedError

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        best_f = 0
        best_x = 0
        for i in range(len(self.results)):
            xi, fi, vi = self.results[i]
            if fi > best_f and vi < SAFETY_THRESHOLD:
                best_x = xi
                best_f = fi

        return best_x
        

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
