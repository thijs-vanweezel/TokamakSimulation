import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, math, torch

def log_normal_diag(x, mu, log_var):
    log_p = -0.5 * keras.ops.log(2. * math.pi) - 0.5 * log_var - 0.5 * keras.ops.exp(-log_var) * (x - mu)**2.
    return log_p

class Forward(keras.Model):
    """
    An eight-layer residual convolutional network with dilated convolutions. Decreases the spatial dimensions by a factor of 16.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        block = lambda filters: keras.Sequential([
            keras.layers.Dropout(.1),
            keras.layers.Conv2D(filters, (3, 3), padding="same"),
            keras.layers.Activation("silu"),
            keras.layers.Conv2D(filters, (3, 3), dilation_rate=(2,2), padding="same"),
            keras.layers.GroupNormalization(groups=-1),
            keras.layers.Activation("silu")    
        ])
        self.block1 = block(64)
        self.pool1 = keras.layers.MaxPooling2D((2, 2))
        self.block2 = block(128)
        self.pool2 = keras.layers.MaxPooling2D((2, 2))
        self.block3 = block(256)
        self.pool3 = keras.layers.MaxPooling2D((2, 2))
        self.block4 = block(512)

    def call(self, x_t):
        h_ = x_t
        h = self.block1(x_t)
        h = keras.layers.add([h, h_]) # residual connection
        h = h_ = self.pool1(h)

        h = self.block2(h)
        h = keras.layers.add([h, h_])
        h = h_ = self.pool2(h)

        h = self.block3(h)
        h = keras.layers.add([h, h_])
        h = h_ = self.pool3(h)

        h = self.block4(h)
        h_t = keras.layers.add([h, h_])
        return h_t

class Prior(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = ...

    @staticmethod
    def reparameterize(mu, log_var):
        eps = keras.random.normal(shape=keras.ops.shape(mu))
        return mu + keras.ops.exp(log_var/2.)*eps

    def call(self, h_t):
        mu_log_var = self.model(h_t)
        mu, log_var = keras.ops.split(mu_log_var, 2, axis=-1)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def log_prob(self, h_t, z):
        _, mu, log_var = self(h_t)
        return log_normal_diag(z, mu, log_var)

class Posterior(keras.Model):
    def __init__(self):
        pass

    def call(self, h_t, x_t_plus1):
        ...
        return z, mu, log_var

    def log_prob(self, z, mu, log_var):
        return log_normal_diag(z, mu, log_var)

class Decoder(keras.Model):
    def __init__(self):
        pass

    def call(self, z, h_t):
        pass

    @staticmethod
    def log_bernoulli(x, p):
        eps = 1.e-5
        pp = keras.ops.clip(p, eps, 1. - eps)
        log_p = x * keras.ops.log(pp) + (1. - x) * keras.ops.log(1. - pp)
        return keras.ops.sum(log_p, list(range(1, keras.ops.ndim(x)))) # sum reduction

    def log_prob(self, x_t_plus1_hat, x_t_plus1):
        return self.log_bernoulli(x_t_plus1, x_t_plus1_hat)

# Instantiate models
forward = Forward()
prior = Prior()
posterior = Posterior()
decoder = Decoder()
opt = keras.optimizers.AdamW()
trainable_weights = forward.trainable_weights + prior.trainable_weights + posterior.trainable_weights + decoder.trainable_weights

# The only part of the code that requires backend-specific ops
def train_step(x_t, x_t_plus1):
    # Forward pass
    h_t = forward(x_t)
    z, mu, log_var = posterior(h_t, x_t_plus1)
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
    gradients = [t.value.grad for t in trainable_weights]
    with torch.no_grad():
        opt.apply_gradients(gradients, trainable_weights)