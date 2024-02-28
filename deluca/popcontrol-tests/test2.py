'''
Example where standard _gpc should do much worse than pop_gpc
(since noise is 0 but want nonzero control)
'''

import jax
import jax.numpy as jnp
import sys
sys.path.append("../agents")
from _pop_gpc import *

def shifted_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    y = x - jnp.array([0.5,0.5])
    return jnp.sum(y.T @ y)

# Define system dynamics matrices
A = jnp.array([[.9, 0], [0.1, 1]])
B = jnp.array([[0.1], [-0.1]])
K = jnp.array([[0,0]])

# Define cost matrices
Q = jnp.array([[1.0, 0], [0, 1.0]])
R = jnp.array([[1.0]])

# Define initial state
initial_state = jnp.array([[0], [1]])

# Initialize GPC agent
gpc_agent = POPGPC(A=A, B=B, Q=Q, R=R, K = K, cost_fn = shifted_loss, HH = 50)
print(gpc_agent.A)

# Define a random key for JAX's random functions
key = jax.random.PRNGKey(0)

# Simulate environment and agent interaction with noise
num_steps = 10000
state = initial_state
tot_loss = 0.0

for step in range(num_steps):
    action = gpc_agent(state)
    tot_loss += gpc_agent.cost_fn(state, action)

    # Update state based on action
    state = A @ state + B @ action

    print(f"Step {step}: Cost = {tot_loss}, State = {state.T}, Action = {action.T}")
