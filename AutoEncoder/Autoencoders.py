import tensorflow.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt

#the encoder, when called, takes the input features and returns the values of the hidden layer (or the bottle neck)
class Encoder(K.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_layer = K.layers.Dense(units=hidden_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return activation


#the sparse encoder add a regularization term (in this case l1 regularization)
class SparseEncoder(K.layers.Layer):
    def __init__(self, hidden_dim, l1_reg=10e-5):
        super().__init__()
        self.hidden_layer = K.layers.Dense(units=hidden_dim, activation=tf.nn.relu, activity_regularizer=tf.keras.regularizers.l1(l1_reg))

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return activation
    

#the decoder, when called, takes the output of the encoder and returns the reconstructed input
class Decoder(K.layers.Layer):
    def __init__(self, original_dim):
        super().__init__()
        self.output_layer = K.layers.Dense(units=original_dim, activation=tf.nn.relu)

    def call(self, encoded):
        activation = self.output_layer(encoded)
        return activation

                    ###########################
                    ### Vanilla Autoencoder ###
                    ###########################

class AutoEncoder(K.Model):
    def __init__(self, hidden_dim, original_dim):
        super().__init__()
        self.loss = []
        self.hidden_dim = hidden_dim
        self.original_dim = original_dim
        self.encoder = Encoder(hidden_dim=self.hidden_dim)
        self.decoder = Decoder(original_dim=self.original_dim)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed

    # define reconstruction loss function
    def calculate_loss(self, preds, real):
        return tf.reduce_mean(tf.square(tf.subtract(preds, real)))

    # define train function for each batch of data
    def train_batch(self, opt, real):
        with tf.GradientTape() as tape:
            preds = self(real)
            error = self.calculate_loss(preds, real)  # reconstruction error
            gradients = tape.gradient(error, self.trainable_variables)
            gradient_variables = zip(gradients, self.trainable_variables)
        opt.apply_gradients(gradient_variables)
        return error

    # train function will be invoked in a loop feeded with the batched dataset
    def train(self, opt, dataset, epochs=20, verbose=1):
        for epoch in range(epochs):
            epoch_loss = 0
            for step, batch_features in enumerate(dataset):
                loss_values = self.train_batch(opt, batch_features)
                epoch_loss += loss_values
            self.loss.append(epoch_loss)
            if verbose == 1:
                print(f'Epoch {epoch + 1}/{epochs}. Loss: {epoch_loss.numpy()}')

    # plot loss vs epochs
    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss');

    def plot_real_vs_reconstructed(self, data, number = 10):
    #### PLOT REAL VS RECONSTRUCTED IMAGES ###
    #number = 10  is the number of image we want to display
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display real
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(data[index].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstructed
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(self(data)[index].numpy().reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


                    ###########################
                    ### Sparse Autoencoder ####
                    ###########################

class SparseAutoEncoder(AutoEncoder):
    def __init__(self, hidden_dim, original_dim):
        super().__init__(hidden_dim, original_dim)
        self.encoder = SparseEncoder(hidden_dim=hidden_dim)
