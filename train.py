import os, math
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch, json
from tqdm.auto import tqdm

def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * keras.ops.log(2. * math.pi) - 0.5 * log_var - 0.5 * keras.ops.exp(-log_var) * (x - mu)**2.
    return log_p

def log_bernoulli(x, p):
    eps = 1.e-5
    pp = keras.ops.clip(p, eps, 1. - eps)
    log_p = x * keras.ops.log(pp) + (1. - x) * keras.ops.log(1. - pp)
    return keras.ops.sum(log_p, list(range(1, keras.ops.ndim(x)))) # sum reduction

# The only function of the code that requires backend-specific ops 
def train_step(x_t, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, opt):
    # Move to gpu
    x_t = x_t.to("cuda")
    x_tplus1 = x_tplus1.to("cuda")
    # Forward pass
    h_t = forward_t(x_t)
    h_tplus1 = forward_tplus1(x_tplus1)
    z, mu, logvar = posterior(h_t, h_tplus1)
    x_tplus1_hat = decoder(z, h_t)
    _, *mu_logvar = prior(h_t)
    kl_nll = keras.ops.sum(
        log_normal_diag(z, mu, logvar) - log_normal_diag(z, *mu_logvar), 
        axis=list(range(1, keras.ops.ndim(z)))  # Sum reduction
    )
    rec_ll = log_bernoulli(x_tplus1, x_tplus1_hat)
    loss = keras.ops.mean(-rec_ll + kl_nll)

    # Prepare backward pass
    forward_t.zero_grad()
    forward_tplus1.zero_grad()
    prior.zero_grad()
    posterior.zero_grad()
    decoder.zero_grad()

    # Backward pass
    loss.backward()
    trainable_weights = forward_t.trainable_weights + forward_tplus1.trainable_weights \
          + prior.trainable_weights + posterior.trainable_weights + decoder.trainable_weights
    gradients = [t.value.grad for t in trainable_weights]
    with torch.no_grad():
        opt.apply_gradients(zip(gradients, trainable_weights))

    # Return loss interpretably
    return x_tplus1_hat, keras.ops.mean(kl_nll).item(), keras.ops.mean(-rec_ll).item()

def run(data, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, n_epochs=100):
    # Loop over epochs
    train_loss_history = {"kl_loss": [0]*n_epochs, "rec_loss": [0]*n_epochs}
    os.makedirs("./results/basic0", exist_ok=True)
    for i in tqdm(range(n_epochs)):
        # Loop over batches
        for j, (x_t, x_tplus1) in enumerate(data, 1):
            # Prepare pushforward training
            if j==1:
                x_t_hat = x_t
            else:
                mask = keras.ops.reshape(data.mask, (-1,1,1,1))
                if False in mask:
                    print("new starting point injected")
                x_t_hat = keras.ops.where(mask, x_t_hat.detach(), x_t[...,:-2])
                x_t_hat = keras.ops.concatenate([x_t_hat, x_t[...,-2:]], axis=-1)
            # Train
            x_t_hat, kl_loss, rec_loss = train_step(x_t_hat, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, optimizer)
            # Just a fancy way to append the mean
            train_loss_history["kl_loss"][i] = train_loss_history["kl_loss"][i]*(1-1/j) + 1/j*kl_loss
            train_loss_history["rec_loss"][i] = train_loss_history["rec_loss"][i]*(1-1/j) + 1/j*rec_loss
        # Save models 
        forward_t.save("./results/basic0/forward_t.keras")
        forward_tplus1.save("./results/basic0/forward_tplus1.keras")
        prior.save("./results/basic0/prior.keras")
        posterior.save("./results/basic0/posterior.keras")
        decoder.save("./results/basic0/decoder.keras")
        # Save training history
        with open("./results/basic0/history.json", "w") as f:
            json.dump(train_loss_history, f)