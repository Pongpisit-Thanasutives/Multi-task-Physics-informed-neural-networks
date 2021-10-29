import torch

def kl_divergence(q, p):
    """Calculates the KL divergence between q and p.

    Tries to compute the KL divergence in closed form, if it
    is not possible, returns the Monte Carlo approximation
    using a single sample.

    Args:
        q : torch.distribution
        Input distribution (posterior approximation).
        p : torch.distribution
        Target distribution (prior).

    Returns: 
        The KL divergence between the two distributions.
    """

    if isinstance(q, torch.distributions.Normal) \
        and isinstance(p, torch.distributions.Normal):

        var_ratio = (q.scale / p.scale.to(q.scale.device)).pow(2)
        t1 = ((q.loc - p.loc.to(q.loc.device)) / p.scale.to(q.loc.device)).pow(2)
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

    else:
        s = q.rsample()
        return q.log_prob(s) - p.log_prob(s)

