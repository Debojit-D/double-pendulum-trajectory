# lnn.py
# -*- coding: utf-8 -*-
"""
Minimal Lagrangian Neural Network utilities for a double pendulum.

Contains:
  - Core equations of motion helpers:
        unconstrained_eom
        lagrangian_eom
        raw_lagrangian_eom
        lagrangian_eom_rk4
        solve_dynamics
  - Simple MLP builder for the Lagrangian:
        mlp(...)
  - custom_init(...) for stable initialization of the MLP weights.

Assumes state = [q1, q2, dq1, dq2] for the double pendulum.
"""

from __future__ import annotations
from functools import partial
from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.example_libraries import stax


# ---------------------------------------------------------------------------
#  Core equations of motion
# ---------------------------------------------------------------------------

def unconstrained_eom(model: Callable, state: jnp.ndarray, t: Any = None) -> jnp.ndarray:
    """
    Generic unconstrained equation of motion:
        state = [q, q_dot]
        model(q, q_dot) -> d/dt [q, q_dot]

    Here `model` is any user-defined function of (q, q_dot)
    that returns a vector of the same size as `state`.
    """
    q, q_dot = jnp.split(state, 2)
    return model(q, q_dot)


def lagrangian_eom(lagrangian: Callable, state: jnp.ndarray, t: Any = None) -> jnp.ndarray:
    """
    Discrete-time style Lagrangian EoM used with small fixed dt.

    - state = [q, q_dot] (size 4 for double pendulum)
    - lagrangian(q, q_dot) -> scalar L(q, q_dot)

    This returns:
        dt * d/dt [q, q_dot] = dt * [q_dot, q_ddot]

    where:
        M(q, q_dot)     = ∂²L / ∂(q_dot)²
        dL/dq           = ∂L / ∂q
        d/dt(∂L/∂q_dot) = J_q (∂L/∂q_dot) · q_dot
    """
    q, q_dot = jnp.split(state, 2)

    # Treat q as angles: wrap to (-π, π] for double pendulum.
    q = (q + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # M(q, q_dot) = ∂²L/∂(q_dot)²
    M = jax.hessian(lagrangian, argnums=1)(q, q_dot)

    # dL/dq
    dL_dq = jax.grad(lagrangian, argnums=0)(q, q_dot)

    # d/dt(∂L/∂q_dot) = J_q(∂L/∂q_dot) @ q_dot
    dLdqdot = jax.grad(lagrangian, argnums=1)
    J_dLdqdot_q = jax.jacobian(dLdqdot, argnums=0)(q, q_dot)
    ddt_dLdqdot = J_dLdqdot_q @ q_dot

    # Solve M q_ddot = dL/dq - d/dt(∂L/∂q_dot)
    # Use pseudo-inverse to avoid singularities
    q_ddot = jnp.linalg.pinv(M) @ (dL_dq - ddt_dLdqdot)

    dt = 1e-1  # small integration step; tune if needed
    return dt * jnp.concatenate([q_dot, q_ddot])


def raw_lagrangian_eom(lagrangian: Callable, state: jnp.ndarray, t: Any = None) -> jnp.ndarray:
    """
    Continuous-time Lagrangian EoM (no dt scaling).

    This is the version typically used in training when your
    targets are the full derivative:
        target = [dq1, dq2, ddq1, ddq2]

    so that:
        preds = raw_lagrangian_eom(lagrangian, state)
    matches that shape.
    """
    q, q_dot = jnp.split(state, 2)

    # Wrap angles for double pendulum.
    q = (q + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # M(q, q_dot) = ∂²L/∂(q_dot)²
    M = jax.hessian(lagrangian, argnums=1)(q, q_dot)

    # dL/dq
    dL_dq = jax.grad(lagrangian, argnums=0)(q, q_dot)

    # d/dt(∂L/∂q_dot)
    dLdqdot = jax.grad(lagrangian, argnums=1)
    J_dLdqdot_q = jax.jacobian(dLdqdot, argnums=0)(q, q_dot)
    ddt_dLdqdot = J_dLdqdot_q @ q_dot

    q_ddot = jnp.linalg.pinv(M) @ (dL_dq - ddt_dLdqdot)
    return jnp.concatenate([q_dot, q_ddot])


def lagrangian_eom_rk4(
    lagrangian: Callable,
    state: jnp.ndarray,
    n_updates: int,
    Dt: float = 1e-1,
    t: Any = None,
) -> jnp.ndarray:
    """
    Single RK4 step for the Lagrangian EoM over a time window Dt,
    subdivided into `n_updates` small sub-steps.

    Returns the increment to `state` over Dt, i.e.:
        state_new = state + lagrangian_eom_rk4(...)
    """

    @jax.jit
    def f(s):
        q, q_dot = jnp.split(s, 2)
        q = (q + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        # Same math as raw_lagrangian_eom but written inline
        M = jax.hessian(lagrangian, argnums=1)(q, q_dot)
        dL_dq = jax.grad(lagrangian, argnums=0)(q, q_dot)
        dLdqdot = jax.grad(lagrangian, argnums=1)
        J_dLdqdot_q = jax.jacobian(dLdqdot, argnums=0)(q, q_dot)
        ddt_dLdqdot = J_dLdqdot_q @ q_dot

        q_ddot = jnp.linalg.pinv(M) @ (dL_dq - ddt_dLdqdot)
        return jnp.concatenate([q_dot, q_ddot])

    @jax.jit
    def one_update(update):
        dt = Dt / float(n_updates)
        s = state + update

        k1 = dt * f(s)
        k2 = dt * f(s + 0.5 * k1)
        k3 = dt * f(s + 0.5 * k2)
        k4 = dt * f(s + k3)

        return update + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    update = jnp.zeros_like(state)
    for _ in range(n_updates):
        update = one_update(update)
    return update


def solve_dynamics(
    dynamics_fn: Callable,
    initial_state: jnp.ndarray,
    is_lagrangian: bool = True,
    **ode_kwargs,
) -> jnp.ndarray:
    """
    Convenience wrapper around `odeint` for either:
      - Lagrangian dynamics (using lagrangian_eom), or
      - a generic unconstrained dynamics function.

    Args:
        dynamics_fn:  If `is_lagrangian` is True, this is a L(q,qdot) function.
                      Otherwise, this is a f(q,qdot) EoM function.
        initial_state:  state vector [q, q_dot]
        is_lagrangian:   choose between lagrangian_eom and unconstrained_eom
        **ode_kwargs:    arguments passed to jax.experimental.ode.odeint
                         (t, rtol, atol, etc.)

    Returns:
        trajectory: array of states over time.
    """
    eom = lagrangian_eom if is_lagrangian else unconstrained_eom

    # Run ODE solve on CPU (typical pattern in GLN code)
    @partial(jax.jit, backend="cpu")
    def _integrate(x0):
        return odeint(partial(eom, dynamics_fn), x0, **ode_kwargs)

    return _integrate(initial_state)


# ---------------------------------------------------------------------------
#  MLP model for L(q, q_dot)
# ---------------------------------------------------------------------------

def mlp(
    args: Any = None,
    input_dim: int | None = None,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    n_hidden_layers: int | None = None,
):
    """
    Build a simple feedforward MLP using JAX stax.

    Usage 1 (with an args object having .hidden_dim and .output_dim):
        init_fun, apply_fun = mlp(args)

    Usage 2 (explicit keyword arguments):
        init_fun, apply_fun = mlp(
            input_dim=4,
            hidden_dim=256,
            output_dim=1,
            n_hidden_layers=3,
        )

    For the double pendulum LNN:
        - input_dim = 4  (q1, q2, dq1, dq2)
        - output_dim = 1 (scalar Lagrangian)
    """
    if args is not None:
        # Legacy style: use args.hidden_dim, args.output_dim
        return stax.serial(
            stax.Dense(args.hidden_dim),
            stax.Softplus,
            stax.Dense(args.hidden_dim),
            stax.Softplus,
            stax.Dense(args.output_dim),
        )

    # Keyword-style
    if hidden_dim is None or output_dim is None:
        raise ValueError("mlp(): must provide hidden_dim and output_dim if args is None.")

    n_layers = n_hidden_layers if n_hidden_layers is not None else 2
    layers = []

    for _ in range(n_layers):
        layers.append(stax.Dense(hidden_dim))
        layers.append(stax.Softplus)

    layers.append(stax.Dense(output_dim))
    return stax.serial(*layers)


# ---------------------------------------------------------------------------
#  Custom initialization for LNN MLP
# ---------------------------------------------------------------------------

def custom_init(init_params, seed: int = 0):
    """
    Apply the "LNN-friendly" initialization from the GLN double pendulum code.

    This assumes:
      - init_params is the parameter tree produced by stax.init_fun
      - The network is a simple uniform-width MLP (all hidden layers same size)
      - You want zero biases + scaled Gaussian weights.

    Returns:
        new_params: same tree structure as init_params.
    """
    import numpy as np

    new_params = []
    rng = jax.random.PRNGKey(seed)

    # Count number of non-empty layers
    number_layers = len([0 for l1 in init_params if len(l1) != 0])
    i = 0

    for l1 in init_params:
        if len(l1) == 0:
            new_params.append(())
            continue

        new_l1 = []
        for l2 in l1:
            if len(l2.shape) == 1:
                # Bias: initialize to zero
                new_l1.append(jnp.zeros_like(l2))
            else:
                # Weight: scaled normal
                n = max(l2.shape)
                first = int(i == 0)
                last = int(i == number_layers - 1)
                mid = int((i != 0) and (i != number_layers - 1))

                std = 1.0 / np.sqrt(n)
                std *= 2.2 * first + 0.58 * mid + n * last

                if std == 0:
                    raise NotImplementedError("custom_init: unexpected layer shape / network structure.")

                new_l1.append(jax.random.normal(rng, l2.shape) * std)
                rng = jax.random.PRNGKey(jax.random.randint(rng, (), 0, 1_000_000))
                i += 1

        new_params.append(new_l1)

    return new_params
