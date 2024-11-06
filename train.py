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
    return keras.ops.mean(log_p, [1,2,3]) # mean reduction

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
    kl_nll = keras.ops.mean(
        log_normal_diag(z, mu, logvar) - log_normal_diag(z, *mu_logvar), 
        axis=[1,2,3]  # mean reduction
    )
    rec_ll = log_bernoulli(x_tplus1, x_tplus1_hat)
    loss = keras.ops.mean(-rec_ll + kl_nll) # mean reduction

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

def val_step(x_t, x_tplus1, forward_t, prior, decoder):
    # Move to gpu
    x_t = x_t.to("cuda")
    x_tplus1 = x_tplus1.to("cuda")
    # Forward pass
    h_t = forward_t(x_t)
    z, *_  = prior(h_t)
    x_tplus1_hat = decoder(z, h_t)
    # Return reconstruction loss
    return keras.ops.mean(-log_bernoulli(x_tplus1, x_tplus1_hat)).item()

def run(train_loader, val_loader, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, save_dir, max_epochs, max_patience=5):
    # Loop over epochs
    patience = 0
    train_loss_history = {"kl_loss": [], "rec_loss": []}
    val_loss_hist = []
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(max_epochs)):
        # Initialize losses
        train_loss_history["kl_loss"].append(0)
        train_loss_history["rec_loss"].append(0)
        # Loop over batches
        for j, (x_t, x_tplus1) in enumerate(train_loader, 1):
            # Prepare pushforward training
            if j==1:
                x_t_hat = x_t
            else:
                mask = keras.ops.reshape(train_loader.dataset.mask, (-1,1,1,1))
                x_t_hat = keras.ops.where(mask, x_t_hat.detach(), x_t[...,:-2]) # detach used in favor of retain_graph
                x_t_hat = keras.ops.concatenate([x_t_hat, x_t[...,-2:]], axis=-1)
            # Train
            x_t_hat, kl_loss, rec_loss = train_step(x_t_hat, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, optimizer)
            # Keep track of losses
            train_loss_history["kl_loss"][i] += kl_loss
            train_loss_history["rec_loss"][i] += rec_loss
        # Normalize losses
        train_loss_history["kl_loss"][i] /= j
        train_loss_history["rec_loss"][i] /= j
        # Validation
        val_loss = 0
        for k, (x_t, x_tplus1) in enumerate(val_loader, 1):
            val_loss += val_step(x_t, x_tplus1, forward_t, prior, decoder)
        val_loss_hist.append(val_loss/k)
        # Early stopping with patience
        if (i>0) and ((val_loss/k)>min(val_loss_hist)):
            patience += 1
            if patience>max_patience:
                break
        else:
            # Reset patience
            patience = 0
            # Save models (keras.saving.save_model causes inconsisteny)
            forward_t.save_weights(f"{save_dir}/forward_t.weights.h5")
            forward_tplus1.save_weights(f"{save_dir}/forward_tplus1.weights.h5")
            prior.save_weights(f"{save_dir}/prior.weights.h5")
            posterior.save_weights(f"{save_dir}/posterior.weights.h5")
            decoder.save_weights(f"{save_dir}/decoder.weights.h5")
            # Save history 
            with open(f"{save_dir}/history.json", "w") as f:
                json.dump(train_loss_history, f)
            with open(f"{save_dir}/val_history.json", "w") as f:
                json.dump(val_loss_hist, f)