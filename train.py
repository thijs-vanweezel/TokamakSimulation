import os, math
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch, json
from tqdm.auto import tqdm
import tensorflow as tf 

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
    PDE loss function compatible with TensorFlow.
    
    Args:
        pred: Prediction of the model x_tplus1_hat
        true: True x_tplus1
        B: Array of b-values (500 constant over instances and time)
        x: Tensor, spatial position variable
        t: Tensor, temporal position variable
    
    Returns:
        f_n: Tensor, computed custom loss term
    """
    pred_n = pred[:, 0]
    pred_v_parallel = pred[:, 1]
    true_n = true[:, 0]
    true_v_parallel = true[:, 1]
    # ## Deel 1 van de formule vgm hoeft dit dus niet anders moeten we t hebben
    # with tf.GradientTape() as tape_t:
    #     tape_t.watch(t)
    #     d_pred_n_dt = tape_t.gradient(pred_n, t)
    #     d_true_n_dt = tape_t.gradient(true_n, t)

    # temporal_loss = tf.square(d_pred_n_dt - d_true_n_dt)

    ## Deel 2 ? In de wor 
    pred_n_v_parallel = pred_n * pred_v_parallel / B
    true_n_v_parallel = true_n * true_v_parallel / B

    with tf.GradientTape() as tape_x:
        tape_x.watch(x)
        d_pred_nv_dx = tape_x.gradient(pred_n_v_parallel, x)
        d_true_nv_dx = tape_x.gradient(true_n_v_parallel, x)

    # Spatial loss component
    spatial_loss = tf.square(B * (d_pred_nv_dx - d_true_nv_dx))

    # Props to chat ligt er dus aan of we dat eerste deel willen 
    # f_n = tf.reduce_mean(temporal_loss) + tf.reduce_mean(spatial_loss)
    return spatial_loss

# The only function of the code that requires backend-specific ops 
def train_step(x_t, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, opt, x_tensor, b_field):
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
        axis=[1,2,3]  # sum reduction
    )
    rec_ll = log_bernoulli(x_tplus1, x_tplus1_hat)
    pde_loss_value = pde_loss(x_tplus1_hat, x_tplus1, b_field, x_tensor)
    loss = keras.ops.mean(-rec_ll + kl_nll) # mean reduction
    loss += pde_loss_value
  
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
    return -log_bernoulli(x_tplus1, x_tplus1_hat)

def run(train_loader, val_loader, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, b_field, x_tensor, save_dir, max_epochs):
    # Loop over epochs
    train_loss_history = {"kl_loss": [0]*max_epochs, "rec_loss": [0]*max_epochs}
    val_loss_hist = []
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(max_epochs)):
        # Loop over batches
        for j, (x_t, x_tplus1) in enumerate(train_loader, 1):
            # Prepare pushforward training
            if j==1:
                x_t_hat = x_t
            else:
                mask = keras.ops.reshape(train_loader.mask, (-1,1,1,1))
                x_t_hat = keras.ops.where(mask, x_t_hat.detach(), x_t[...,:-2]) # detach used in favor of retain_graph
                x_t_hat = keras.ops.concatenate([x_t_hat, x_t[...,-2:]], axis=-1)
            # Train
            x_t_hat, kl_loss, rec_loss = train_step(x_t_hat, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, x_tensor, b_field)
            # Just a fancy way to append the mean
            train_loss_history["kl_loss"][i] = train_loss_history["kl_loss"][i]*(1-1/j) + 1/j*kl_loss
            train_loss_history["rec_loss"][i] = train_loss_history["rec_loss"][i]*(1-1/j) + 1/j*rec_loss
        # Validation
        val_loss = 0
        for k, (x_t, x_tplus1) in enumerate(val_loader, 1):
            val_loss += val_step(x_t, x_tplus1, forward_t, prior, decoder)
            val_loss_hist.append(val_loss/k)
        if len(val_loss_hist) > 1 and val_loss > val_loss_hist[-2]:  # Compare with the previous epoch’s validation loss
            break
        # Save models 
        forward_t.save(f"{save_dir}/forward_t.keras")
        forward_tplus1.save(f"{save_dir}/forward_tplus1.keras")
        prior.save(f"{save_dir}/prior.keras")
        posterior.save(f"{save_dir}/posterior.keras")
        decoder.save(f"{save_dir}/decoder.keras")
        # Save training history (trimmed to exclude default zeros)

        trimmed_train_loss_history = {key: [val for val in values if val != 0] for key, values in train_loss_history.items()}
        # Save the trimmed history to a JSON file
        with open(f"{save_dir}/history.json", "w") as f:
            json.dump(trimmed_train_loss_history, f)
        # with open(f"{save_dir}/history.json", "w") as f:
        #     json.dump(train_loss_history[:train_loss_history.index(0)], f)