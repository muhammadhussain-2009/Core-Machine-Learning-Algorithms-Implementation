#  Wasserstein Critic Loss for Wasserstein GANs (WGAN). In WGANs, the discriminator is replaced with a critic that outputs real-valued scores instead of probabilities.
# Wasserstein Critic Loss: L=E[D(fake)]−E[D(real)]

import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores= np.asarray(real_scores)
    fake_scores= np.asarray(fake_scores)
    if not isinstance(real_scores, np.ndarray):
        raise TypeError("Incorrect Array Type")
    if not isinstance(fake_scores, np.ndarray):
        raise TypeError("Incorrect Array Type")
    loss = np.mean(fake_scores)- np.mean(real_scores)
    return float(loss)
    pass
