import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch
from models import Forward, Prior, Posterior, Decoder

# Instantiate models
forward = Forward()
prior = Prior()
posterior = Posterior()
decoder = Decoder()
opt = keras.optimizers.AdamW()

# The only part of the code that requires backend-specific ops 
def train_step(x_t, x_t_plus1):
    # Forward pass
    h_t = forward(x_t)
    h_tplus1 = forward(x_t_plus1)
    z, mu, log_var = posterior(h_t, h_tplus1)
    kl_loss = posterior.log_prob(z, mu, log_var) - prior.log_prob(h_t, z)
    rec_loss = decoder.log_prob(x_t_plus1, decoder(z, h_t))
    loss = keras.ops.mean(-(rec_loss - kl_loss))

    # Prepare backward pass
    forward.zero_grad()
    prior.zero_grad()
    posterior.zero_grad()
    decoder.zero_grad()

    # Backward pass
    loss.backward()
    trainable_weights = forward.trainable_weights + prior.trainable_weights + posterior.trainable_weights + decoder.trainable_weights
    gradients = [t.value.grad for t in trainable_weights]
    with torch.no_grad():
        opt.apply_gradients(zip(gradients, trainable_weights))