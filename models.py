import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, math

def reparameterize(mu, log_var):
    eps = keras.random.normal(shape=keras.ops.shape(mu))
    return mu + keras.ops.exp(log_var/2.)*eps

class Forward(keras.Model):
    """
    An eight-layer residual convolutional network with dilated convolutions. Decreases the spatial dimensions by a factor of 8.
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
        self.conv1x1_1 = keras.layers.Conv2D(32, (1, 1), padding="same")
        self.block1 = block(32)
        self.pool1 = keras.layers.MaxPooling2D((1, 2))
        self.conv1x1_2 = keras.layers.Conv2D(64, (1, 1), padding="same")
        self.block2 = block(64)
        self.pool2 = keras.layers.MaxPooling2D((1, 2))
        self.conv1x1_3 = keras.layers.Conv2D(128, (1, 1), padding="same")
        self.block3 = block(128)
        self.pool3 = keras.layers.MaxPooling2D((1, 2))
        self.block4 = block(256)

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
        h = self.block3(h)
        h = keras.layers.add([h, h_])
        h = self.pool3(h)

        h_t = self.block4(h)
        return h_t

class Prior(keras.Model):
    """
    Takes an input and returns a normal distribution and a sample of equal size.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            keras.layers.Conv2D(512, (3, 3), padding="same"),
            keras.layers.Activation("silu"),
            keras.layers.Conv2DTranspose(512, (3, 3), padding="same"),
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
            keras.layers.Conv2D(512, (3, 3), padding="same"),
            keras.layers.Activation("silu"),
            keras.layers.Conv2DTranspose(512, (3, 3), padding="same"),
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
        block = lambda filters, activation="silu", strides=None: keras.Sequential([
            keras.layers.Dropout(.1),
            keras.layers.Conv2DTranspose(filters, (3, 3), padding="same"),
            keras.layers.Activation("silu"),
            keras.layers.Conv2DTranspose(filters, (3, 3), strides=strides if strides else (1, 2), padding="same"),
            keras.layers.GroupNormalization(groups=-1),
            keras.layers.Activation(activation) 
        ])
        self.conv1x1_1 = keras.layers.Conv2DTranspose(256, (1, 1), strides=(1, 2), padding="same")
        self.block1 = block(256)
        self.padding = keras.layers.ZeroPadding2D(((0, 0), (1, 0)))
        self.conv1x1_2 = keras.layers.Conv2DTranspose(128, (1, 1), strides=(1, 2), padding="same")
        self.block2 = block(128)
        self.conv1x1_3 = keras.layers.Conv2DTranspose(64, (1, 1), strides=(1, 2), padding="same")
        self.block3 = block(64)
        self.block4 = block(6, "linear", (1, 1))

    def call(self, z, h_t):
        x = keras.layers.concatenate([z, h_t])

        x_ = self.conv1x1_1(x)
        x = self.block1(x)
        x = keras.layers.add([x, x_]) # residual connection

        x = self.padding(x) # Required for exact shape matching
        x_ = self.conv1x1_2(x)
        x = self.block2(x)
        x = keras.layers.add([x, x_])

        x_ = self.conv1x1_3(x)
        x = self.block3(x)
        x = keras.layers.add([x, x_])

        x_tplus1_hat = self.block4(x)
        return keras.ops.sigmoid(x_tplus1_hat)