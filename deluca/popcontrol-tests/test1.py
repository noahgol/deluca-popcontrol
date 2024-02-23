import jax
import jax.numpy as jnp
import sys

sys.path.append('/Users/ngolowich/Documents/mit/rl/population-control/popcontrol-code/deluca/agents')
from _gpc import GPC, quad_loss

# Define system dynamics matrices
A = jnp.array([[1.0, 0.1], [0, 1.0]])
B = jnp.array([[0], [0.1]])

# Define cost matrices
Q = jnp.array([[1.0, 0], [0, 1.0]])
R = jnp.array([[1.0]])

# Define initial state
initial_state = jnp.array([[0], [0]])

# Initialize GPC agent
gpc_agent = GPC(A=A, B=B, Q=Q, R=R)

# Define a random key for JAX's random functions
key = jax.random.PRNGKey(0)

# Simulate environment and agent interaction with noise
num_steps = 10
state = initial_state

for step in range(num_steps):
    action = gpc_agent(state)
    
    # Generate random noise for the state update
    noise = jax.random.normal(key, shape=(2, 1)) * 0.05  # Adjust the scale as necessary
    key, _ = jax.random.split(key)  # Update the key for next usage
    
    # Update state based on action and add noise
    state = A @ state + B @ action + noise
    
    print(f"Step {step}: State = {state.T}, Action = {action.T}, Noise = {noise.T}")
