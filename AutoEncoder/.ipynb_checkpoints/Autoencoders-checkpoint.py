import tensorflow.keras as K
import tensorflow as tf

#the encoder, when called, takes the input features and returns the values of the hidden layer (or the bottle neck)
class Encoder(K.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_layer = K.layers.Dense(units=hidden_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return activation(self)

#the decoder, when called, takes the output of the encoder and returns the reconstructed input
class Decoder(K.layers.Layer):
    def __init__(self, original_dim):
        super().__init__()
        self.output_layer = K.layers.Dense(units=original_dim, activation=tf.nn.relu)

    def call(selfself, encoded):
        activation = self.output_layer(encoded)
        return(activation)

###########################
### Vanilla Autoencoder ###
###########################

class Autoencoder(k.Model):
    def __init__(self, hidden_dim, original_dim):
        super().__init__()
        self.loss = []
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed