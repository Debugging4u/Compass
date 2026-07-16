import pandas as pd
import numpy as np
from scipy import optimize

#exercise problem 4
def calculate_optimal_weights(mu: np.ndarray, cov_matrix: np.ndarray, risk_aversion: float) -> np.ndarray:
    """
    Function that calculates optimal port. weights

    Parameters
    ----------
    mu:
        Expected returns
    cov_matrix:
        Covariance matrix
    risk_aversion:
        Risk aversion parameter

    Returns
    -------
    float
        Optimal portfolio weights
    """
    sigma_inv = np.linalg.inv(cov_matrix)
    A = np.ones_like(mu)
    C = A @ sigma_inv @ A
    C_inv = 1.0 / C
    b = 1.0

    first_part = 1.0 / risk_aversion * sigma_inv @ mu
    second_part = 1.0 / risk_aversion * sigma_inv @ A * C_inv * (A @ sigma_inv @ mu - risk_aversion * b)

    opt_weights = first_part - second_part

    return opt_weights



#exercise problem 4
def portfolio_optimization(mu_est, cov_mat_est, target_return=None, no_short_selling=True, display=False):
    """
    Generalized function for portfolio optimization.

    Parameters:
    -----------
    mu_est: np.ndarray
        Expected returns of the assets.
    cov_mat_est: np.ndarray
        Covariance matrix of the assets.
    target_return: float, optional
        The target return for the portfolio. If None, the optimization will minimize variance without a return constraint.
    no_short_selling: bool, optional
        If True, short selling is prohibited (i.e., all portfolio weights must be non-negative).
    display: bool, optional
        If True, displays detailed output from the optimization process.

    Returns:
    --------
    result: OptimizeResult
        The result of the optimization, including the optimized portfolio weights and additional information.
    """

    # Define the objective function (portfolio variance)
    def portfolio_variance(weights, cov_mat):
        return weights.T @ cov_mat @ weights

    # Define the derivative of the objective function (Jacobian)
    def portfolio_variance_derivative(weights, cov_mat):
        return 2 * cov_mat @ weights

    # Constraint: The sum of the portfolio weights must be 1 (fully invested)
    sum_to_one_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones_like(x)}

    # Constraint: Target return constraint if provided
    if target_return is not None:
        target_return_constraint = {
            'type': 'eq',
            'fun': lambda x: x @ mu_est - target_return,
            'jac': lambda x: mu_est
        }
    else:
        target_return_constraint = None

    # Constraint: No short selling (weights must be non-negative)
    if no_short_selling:
        no_short_constraint = {'type': 'ineq', 'fun': lambda x: x, 'jac': lambda x: np.eye(len(x))}
    else:
        no_short_constraint = None

    # Set up constraints list
    constraints = [sum_to_one_constraint]
    if target_return_constraint:
        constraints.append(target_return_constraint)
    if no_short_constraint:
        constraints.append(no_short_constraint)

    # Initial guess (equal weights)
    initial_guess = np.ones(len(mu_est)) / len(mu_est)

    # Run optimization
    result = optimize.minimize(portfolio_variance, x0=initial_guess, args=(cov_mat_est,),
                               method='SLSQP',
                               jac=portfolio_variance_derivative,
                               constraints=constraints,
                               options={'disp': display})

    return result

# Example usage:
# mu_est = np.array([0.02, 0.03, 0.015]) # Example expected returns
# cov_mat_est = np.array([[0.0004, 0.0002, 0.0001], [0.0002, 0.0003, 0.00015], [0.0001, 0.00015, 0.0002]]) # Example covariance matrix
# target_return = 0.025
# result = portfolio_optimization(mu_est, cov_mat_est, target_return=target_return, no_short_selling=True)
# print(result.x) # Optimized weights
