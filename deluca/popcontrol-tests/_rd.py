import sys
sys.path.append('..')
from core import Env
import jax
import jax.numpy as jnp


class RD(Env):

    def __init__(self, A, eta):
        assert eta > 0, "eta must be greater than 0"
        self.A = A
        self.eta = eta
        self.state = None

    def init(self, initial_state=None):
        """Initialize or reset the internal state x."""
        d = self.A.shape[0]  # Dimension of the matrix A, assuming A is dxd
        if initial_state is None:
            # Initialize x as a uniform distribution if no initial state is provided
            self.state = jnp.ones(d) / d
        else:
            # Use the provided initial state
            self.state = initial_state / jnp.sum(initial_state)
        return self.state

    def __call__(self, U):
        """Update the internal state x according to the specified dynamics with matrix U."""
        # Compute (A + U) * x
        AUx = (self.A + U) @ self.state
        # Update x according to the given rule
        x_prime = self.state * (1 + self.eta * AUx)
        C = jnp.sum(x_prime)  # Normalizing constant
        self.state = x_prime / C
        return self.state

# Example usage:
d = 3  # Dimension of the matrix A and vector x
A = jax.random.uniform(jax.random.PRNGKey(0), (d, d))
eta = 0.01

rd = RD(A, eta)
initial_state = jax.random.uniform(jax.random.PRNGKey(1), (d,))
rd.init(initial_state=initial_state)

# Example call to update the state with U
U = jax.random.uniform(jax.random.PRNGKey(2), (d, d))
print("Initial state:", rd.state)
new_state = rd(U)
print("A+U:", A+U)
print("Updated state:", rd.state)


