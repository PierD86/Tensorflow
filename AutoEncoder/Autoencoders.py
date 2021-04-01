import tensorflow.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D

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
    
#the encoder, for Convolutional Autoencoder, consists in 3 convolutional layers, each followed by a max pooling layer
class ConvEncoder(K.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.pool = MaxPooling2D((2,2), padding='same')

    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)        
        return x

#the decoder, is the opposite of encoder. We are using UpSampling as dual of MaxPooling to increase the size back
class ConvDecoder(K.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation='relu', padding='valid')
        self.conv4 = Conv2D(1, kernel_size=3, strides=1, activation='sigmoid', padding='same')
        self.upsample = UpSampling2D((2,2))

    def call(self, encoded):
        x = self.conv1(encoded)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)        
        return self.conv4(x)


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

    
                    ###########################
                    #Convolutional Autoencoder#
                    ###########################

class ConvAutoEncoder(AutoEncoder):
    def __init__(self, filters):
        K.Model.__init__(self)
        self.loss = []
        self.filters = filters
        self.encoder = ConvEncoder(filters=self.filters)
        self.decoder = ConvDecoder(filters=self.filters)