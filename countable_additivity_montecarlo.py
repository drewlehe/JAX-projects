import jax
import jax.numpy as jnp

# -------- 1. Generate the first N disjoint intervals A_n --------
def generate_intervals(N: int):
    """
    Generate intervals A_n = (1 - 2^{-n+1}, 1 - 2^{-n}) for n=1,...,N.
    Returns shape (N, 2): [left, right].
    """
    n = jnp.arange(1, N + 1)
    lefts = 1.0 - 2.0 ** (-(n - 1))
    rights = 1.0 - 2.0 ** (-n)
    return jnp.stack([lefts, rights], axis=1)  # (N, 2)


# -------- 2. Indicator for union of intervals --------
def indicator_union(x, intervals):
    """
    x: scalar in [0, 1]
    intervals: array of shape (N, 2) with [left, right] for each interval

    Returns 1.0 if x is in the union of intervals, else 0.0.
    """
    lefts = intervals[:, 0]
    rights = intervals[:, 1]
    inside_any = jnp.any((lefts < x) & (x < rights))
    return jnp.where(inside_any, 1.0, 0.0)


indicator_union_vec = jax.vmap(indicator_union, in_axes=(0, None))


# -------- 3. Monte Carlo estimator for the measure of the union --------
def mc_measure_union(key, intervals, num_samples=200_000):
    """
    Approximate Lebesgue measure of union of intervals within [0,1]
    by Monte Carlo with Uniform(0,1) samples.
    """
    xs = jax.random.uniform(key, shape=(num_samples,), minval=0.0, maxval=1.0)
    vals = indicator_union_vec(xs, intervals)
    # Since domain length is 1, mean of indicator is the measure
    return jnp.mean(vals)


# -------- 4. Put it all together for several N --------
key = jax.random.PRNGKey(0)

for N in [1, 2, 3, 5, 10, 20]:
    intervals = generate_intervals(N)

    # True measure by countable additivity (finite version: sum of lengths)
    lengths = intervals[:, 1] - intervals[:, 0]
    true_measure = jnp.sum(lengths)

    # Monte Carlo estimate
    key, subkey = jax.random.split(key)
    mc_est = mc_measure_union(subkey, intervals, num_samples=100_000)

    print(f"N = {N}")
    print(f"  Sum of measures (finite additivity): {float(true_measure):.6f}")
    print(f"  Monte Carlo estimate of union:       {float(mc_est):.6f}")
    print()
