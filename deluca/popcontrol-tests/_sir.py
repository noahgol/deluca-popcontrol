import sys
sys.path.append('..')
from core import Env
import jax
import jax.numpy as jnp

class SIR(Env):

    def __init__(self, beta, gamma, delta_t=0.01):
        """
        Initializes the SIR model parameters.
        
        Args:
            beta (float): Transmission rate per contact per day.
            gamma (float): Recovery rate per day.
            delta_t (float): Time step for the discrete update.
        """
        self.beta = beta
        self.gamma = gamma
        self.delta_t = delta_t

    def init(self, initial_state):
        """
        Initialize or reset the internal state [S, I, R].
        """
        assert initial_state.shape == (3,), "State must be a 3-dimensional vector."
        assert jnp.sum(initial_state) == 1, "The sum of S, I, and R proportions must equal 1."
        self.state = initial_state
        return self.state

    def __call__(self, state, action=None, *args, **kwargs):
        """
        Update the state based on the SIR dynamics.
        """
        S, I, R = state
        new_S = S - self.beta * S * I * self.delta_t
        new_I = I + (self.beta * S * I - self.gamma * I) * self.delta_t
        new_R = R + self.gamma * I * self.delta_t

        # Ensure the state remains in valid bounds
        new_state = jnp.array([new_S, new_I, new_R])
        new_state = jnp.clip(new_state, 0, 1)
        new_state /= jnp.sum(new_state)  # Normalize to ensure the sum is 1

        return new_state

# Example usage
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
initial_state = jnp.array([0.99, 0.01, 0.0])  # Almost everyone is susceptible

sir_model = SIR(beta, gamma)
sir_model.init(initial_state)

# Simulate one update step
print("Initial state:", initial_state)
new_state = sir_model(initial_state)
print("Updated state:", new_state)
