import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch, json
from tqdm.auto import tqdm

# The only function of the code that requires backend-specific ops 
def train_step(x_t, x_t_plus1, forward_t, forward_tplus1, prior, posterior, decoder, opt):
    # Move to gpu
    x_t = x_t.to("cuda")
    x_t_plus1 = x_t_plus1.to("cuda")
    # Forward pass
    h_t = forward_t(x_t)
    h_tplus1 = forward_tplus1(x_t_plus1)
    z, mu, log_var = posterior(h_t, h_tplus1)
    kl_nll = keras.ops.sum(posterior.log_prob(z, mu, log_var) - prior.log_prob(h_t, z), axis=(1,2,3)) # Sum reduction
    rec_ll = decoder.log_prob(x_t_plus1, decoder(z, h_t))
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

    return keras.ops.mean(kl_nll).item(), keras.ops.mean(-rec_ll).item()

def run(dataloader, forward_t, forward_tplus1, prior, posterior, decoder, optimizer, n_epochs=100):
    # Loop over epochs
    train_loss_history = {"kl_loss": [], "rec_loss": []}
    os.makedirs("./results/basic0", exist_ok=True)
    for i in tqdm(range(n_epochs)):
        # Loop over batches
        for j, (x_t, x_tplus1) in enumerate(dataloader, 1):
            # Train
            kl_loss, rec_loss = train_step(x_t, x_tplus1, forward_t, forward_tplus1, prior, posterior, decoder, optimizer)
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