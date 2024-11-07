import os, math
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch, json
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * keras.ops.log(2. * math.pi) - 0.5 * log_var - 0.5 * keras.ops.exp(-log_var) * (x - mu)**2.
    return log_p

# The only function of the code that requires backend-specific ops 
def train_step(x_t, x_tplus1, forward_t, decoder, opt):
    # Move to gpu
    x_t = x_t.to(DEVICE)
    x_tplus1 = x_tplus1.to(DEVICE)
    # Forward pass
    h_t = forward_t(x_t)
    x_tplus1_hat = decoder(h_t)
    loss = keras.ops.mean(keras.ops.square(x_tplus1 - x_tplus1_hat)) # mean (squared) reduction

    # Prepare backward pass
    forward_t.zero_grad()
    decoder.zero_grad()

    # Backward pass
    loss.backward()
    trainable_weights = forward_t.trainable_weights + decoder.trainable_weights
    gradients = [t.value.grad for t in trainable_weights]
    with torch.no_grad():
        opt.apply_gradients(zip(gradients, trainable_weights))

    # Return loss interpretably
    return x_tplus1_hat, loss.item()

def val_step(x_t, x_tplus1, forward_t, decoder):
    # Move to gpu
    x_t = x_t.to(DEVICE)
    x_tplus1 = x_tplus1.to(DEVICE)
    # Forward pass
    h_t = forward_t(x_t)
    x_tplus1_hat = decoder(h_t)
    # Return reconstruction loss
    rec_nl = keras.ops.mean(keras.ops.square(x_tplus1 - x_tplus1_hat)) # mean (squared) reduction
    return rec_nl.item()

def run(train_loader, val_loader, forward_t, decoder, optimizer, save_dir, max_epochs, max_patience=5):
    # Loop over epochs
    patience = 0
    loss_history = {"rec_loss": [], "val_rec_loss": []}
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(max_epochs)):
        # Initialize losses
        loss_history["rec_loss"].append(0)
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
            x_t_hat, rec_loss = train_step(x_t_hat, x_tplus1, forward_t, decoder, optimizer)
            # Keep track of losses
            loss_history["rec_loss"][i] += rec_loss
        # Normalize losses
        loss_history["rec_loss"][i] /= j
        # Validation
        val_loss = 0
        for k, (x_t, x_tplus1) in enumerate(val_loader, 1):
            val_loss += val_step(x_t, x_tplus1, forward_t, decoder)
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
            decoder.save_weights(f"{save_dir}/decoder.weights.h5")
            # Save history 
            with open(f"{save_dir}/history.json", "w") as f:
                json.dump(loss_history, f)