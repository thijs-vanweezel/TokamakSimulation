import os, math
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch, json
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * keras.ops.log(2. * math.pi) - 0.5 * log_var - 0.5 * keras.ops.exp(-log_var) * (x - mu)**2.
    return log_p

def log_bernoulli(x, p):
    eps = 1.e-5
    pp = keras.ops.clip(p, eps, 1. - eps)
    log_p = x * keras.ops.log(pp) + (1. - x) * keras.ops.log(1. - pp)
    return keras.ops.sum(log_p, [1,2,3]) # sum reduction

def pde_loss(pred, true, B, x):
    """
    PDE loss function in PyTorch.
    
    Args:
        pred: Tensor, prediction of the model (x_tplus1_hat), with shape (batch_size, num_features)
        true: Tensor, true values (x_tplus1), with shape (batch_size, num_features)
        B: Tensor or scalar, magnetic field strength, assumed constant over instances and time
        x: Tensor, spatial position variable, assumed to be in shape (batch_size,)
    
    Returns:
        spatial_loss: Tensor, computed spatial loss term
    """
    # Extract predicted and true components
    pred_n = pred[:, 0]
    pred_v_parallel = pred[:, 1]
    true_n = true[:, 0]
    true_v_parallel = true[:, 1]
    
    # Compute (predicted) n * v_parallel / B and (true) n * v_parallel / B
    pred_n_v_parallel = (pred_n * pred_v_parallel) / B
    true_n_v_parallel = (true_n * true_v_parallel) / B

    # Compute spatial derivatives with respect to x
    d_pred_nv_dx = torch.autograd.grad(pred_n_v_parallel, x, grad_outputs=torch.ones_like(pred_n_v_parallel), create_graph=True)[0]
    d_true_nv_dx = torch.autograd.grad(true_n_v_parallel, x, grad_outputs=torch.ones_like(true_n_v_parallel), create_graph=True)[0]

    # Spatial loss component
    spatial_loss = torch.square(B * (d_pred_nv_dx - d_true_nv_dx))

    # Return the mean spatial loss across the batch
    return spatial_loss.mean()

# The only function of the code that requires backend-specific ops 
def train_step(x_t, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, opt, x_tensor, b_field):
    # Move to gpu
    x_t = x_t.to(DEVICE)
    x_tplus1 = x_tplus1.to(DEVICE)
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
    rec_nl = keras.ops.mean( # MSE == \mathcal{N}(\eps|0,1)
        keras.ops.square(x_tplus1 - x_tplus1_hat),
        axis=[1,2,3]  # mean reduction
    )
    pde_loss_value = pde_loss(x_tplus1_hat, x_tplus1, b_field, x_tensor)
    loss = keras.ops.mean(rec_nl + kl_nll)+pde_loss_value # mean reduction
  
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
    return x_tplus1_hat, keras.ops.mean(kl_nll).item(), keras.ops.mean(rec_nl).item(), pde_loss_value.item()

def val_step(x_t, x_tplus1, forward_t, prior, decoder):
    # Move to gpu
    x_t = x_t.to(DEVICE)
    x_tplus1 = x_tplus1.to(DEVICE)
    # Forward pass
    h_t = forward_t(x_t)
    z, *_  = prior(h_t)
    x_tplus1_hat = decoder(z, h_t)
    # Return reconstruction loss
    rec_nl = keras.ops.mean(keras.ops.square(x_tplus1 - x_tplus1_hat)) # mean (squared) reduction
    return rec_nl.item()

def run(train_loader, val_loader, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, b_field, x_tensor, save_dir, max_epochs, max_patience=5):
    # Loop over epochs
    patience = 0
    loss_history = {"kl_loss": [], "rec_loss": [], "val_rec_loss": [], "pde_loss": []}
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(max_epochs)):
        # Initialize losses
        loss_history["kl_loss"].append(0)
        loss_history["rec_loss"].append(0)
        loss_history["pde_loss"].append(0)
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
            x_t_hat, kl_loss, rec_loss, pde_loss = train_step(x_t_hat, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, x_tensor, b_field)
            # Keep track of losses
            loss_history["kl_loss"][i] += kl_loss
            loss_history["rec_loss"][i] += rec_loss
            loss_history["pde_loss"][i] += pde_loss
        # Normalize losses
        loss_history["kl_loss"][i] /= j
        loss_history["rec_loss"][i] /= j
        loss_history["pde_loss"][i] /= j
        # Validation
        val_loss = 0
        for k, (x_t, x_tplus1) in enumerate(val_loader, 1):
            val_loss += val_step(x_t, x_tplus1, forward_t, prior, decoder)
        loss_history["val_rec_loss"].append(val_loss/k)
        # Early stopping with patience
        if (i>0) and ((val_loss/k)>min(loss_history["val_rec_loss"])):
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
                json.dump(loss_history, f)
