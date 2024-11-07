import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, math

def reparameterize(mu, log_var):
    eps = keras.random.normal(shape=keras.ops.shape(mu))
    return mu + keras.ops.exp(log_var/2.)*eps

class Forward(keras.Model):
    """
    An eight-layer residual convolutional network with dilated convolutions. Removes the spatial dimension.
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

        self.conv1x1_1 = keras.layers.Conv2D(16, (1, 1), padding="same")
        self.block1 = block(16)
        self.pool1 = keras.layers.MaxPooling2D((3, 3))
        self.conv1x1_2 = keras.layers.Conv2D(32, (1, 1), padding="same")
        self.block2 = block(32)
        self.pool2 = keras.layers.MaxPooling2D((3, 3))
        self.conv1x1_3 = keras.layers.Conv2D(64, (1, 1), padding="same")

        self.block_q = block(64)
        self.block_v = block(64)
        self.block_k = block(64)
        self.attention = keras.layers.Attention()
        self.norm = keras.layers.LayerNormalization()
        self.globalpool = keras.layers.GlobalMaxPooling2D()
        self.dense = keras.layers.Dense(512)

    def call(self, x_t):
        h_ = self.conv1x1_1(x_t)
        h = self.block1(x_t)
        h = keras.layers.add([h, h_]) # residual connection
        h = self.pool1(h)

        h_ = self.conv1x1_2(h)
        h = self.block2(h)
        h = keras.layers.add([h, h_])
        h = self.pool2(h)

        h_ = self.conv1x1_3(h)
        q = self.block_q(h)
        b,he,w,c = keras.ops.shape(q)
        q = keras.ops.reshape(q, (b,he*w,c))
        v = self.block_v(h)
        v = keras.ops.reshape(v, (b,he*w,c))
        k = self.block_k(h)
        k = keras.ops.reshape(k, (b,he*w,c))
        h = self.attention([q, v, k])
        h = keras.ops.reshape(h, (b,he,w,c))
        h = keras.layers.add([h, h_])
        h = self.norm(h)
        h = self.globalpool(h)
        h_t = self.dense(h)
        return h_t

class Prior(keras.Model):
    """
    Takes an input and returns a normal distribution and a sample of equal size.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            keras.layers.Dense(512),
            keras.layers.Activation("silu"),
            keras.layers.Dense(1000)
        ])

    def call(self, h_t):
        mu_logvar = self.net(h_t)
        mu, logvar = keras.ops.split(mu_logvar, 2, axis=-1)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Posterior(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            keras.layers.Dense(512),
            keras.layers.Activation("silu"),
            keras.layers.Dense(1024)
        ])

    def call(self, h_t, x_tplus1):
        # Assume h_t and x_tplus1 have equal dimensions
        c = keras.layers.concatenate([h_t, x_tplus1])
        mu_logvar = self.net(c)
        mu, logvar = keras.ops.split(mu_logvar, 2, axis=-1)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        block = lambda filters, activation="silu": keras.Sequential([
            keras.layers.Dropout(.1),
            keras.layers.Conv1DTranspose(filters, 3, padding="same"),
            keras.layers.Activation("silu"),
            keras.layers.Conv1DTranspose(filters, 3, padding="same"),
            keras.layers.GroupNormalization(groups=-1),
            keras.layers.Activation(activation) 
        ])
        self.dense = keras.layers.Dense(500)
        self.reshape = keras.layers.Reshape((500, 1))
        self.conv1x1_1 = keras.layers.Conv1D(64, 1, padding="same")
        self.block1 = block(64)
        self.conv1x1_2 = keras.layers.Conv1D(128, 1, padding="same")
        self.block2 = block(128)
        self.conv1x1_3 = keras.layers.Conv1D(64, 1, padding="same")
        self.block3 = block(64)
        self.block4 = block(6, "linear")

    def call(self, z, h_t):
        x = keras.layers.concatenate([z, h_t])
        x = self.dense(x)
        x = self.reshape(x)

        x_ = self.conv1x1_1(x)
        x = self.block1(x)
        x = keras.layers.add([x, x_]) # residual connection

        x_ = self.conv1x1_2(x)
        x = self.block2(x)
        x = keras.layers.add([x, x_])

        x_ = self.conv1x1_3(x)
        x = self.block3(x)
        x = keras.layers.add([x, x_])

        x_tplus1_hat = self.block4(x)
        return x_tplus1_hat