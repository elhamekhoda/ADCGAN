import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import glob
import h5py
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler,StandardScaler



def build_generator_model():
    model = tf.keras.Sequential(name = "Generator")

    model.add(tf.keras.Input(shape = (GEN_DIM,)))
    model.add(layers.Dense(128, kernel_initializer = 'glorot_uniform'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Reshape((8, 8, 2)))

    model.add(layers.Conv2DTranspose(32, kernel_size = 2, strides = 1, padding = "same"))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(16, kernel_size = 3, strides = 1, padding = "same"))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(len(train_features), activation = 'tanh'))

    return model

def build_generator_model_dnn():
    model = tf.keras.Sequential(name = "Generator")

    model.add(tf.keras.Input(shape = (GEN_DIM,)))
    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="gen_layer1", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="gen_layer2", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="gen_layer3", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="gen_layer4", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(len(train_features), activation = 'tanh'))

    return model

def build_discriminator_model():
    model = tf.keras.Sequential(name = "Discriminator")

    model.add(tf.keras.Input(shape = (DISC_DIM,)))
    model.add(layers.Dense(128))
    model.add(layers.Reshape((8, 8, 2)))

    model.add(layers.Conv2D(64, kernel_size = 3, strides = 1, padding = "same"))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Conv2D(32, kernel_size = 3, strides = 1, padding = "same"))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Conv2D(16, kernel_size = 3, strides = 1, padding = "same"))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Flatten())
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    opt = Adam(lr=0.005, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_discriminator_model_dnn():
    model = tf.keras.Sequential(name = "Discriminator")

    model.add(tf.keras.Input(shape = (DISC_DIM,)))

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="disc_layer1", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="disc_layer2", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="disc_layer3", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Dense(256, kernel_initializer = 'glorot_uniform', name="disc_layer4", kernel_regularizer=regularizers.L2(0.0001)))
    model.add(layers.LeakyReLU(alpha = 0.2))

    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    opt = Adam(lr=0.0003, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def create_gan(discriminator, generator, latent_dim):
    discriminator.trainable=False
    gan_input = tf.keras.Input(shape=(latent_dim,), name="Noise")
    cond_input = tf.keras.Input(shape=(1,), name="Condition")
    x = generator(layers.Concatenate(axis=-1, name="GenInput")([gan_input,cond_input]))
    gan_output= discriminator(layers.Concatenate(axis=-1, name="DiscInput")([x,cond_input]))
    gan= tf.keras.Model(inputs=[gan_input,cond_input], outputs=gan_output, name="GAN")
    gen_opt = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=gen_opt)
    return gan


# select real samples
def generate_real_samples(train_dataset, cond_data, n_samples):
    # choose random instances
    ix = np.random.randint(0, train_dataset.shape[0], n_samples)
    # select images
    X = train_dataset[ix]
    # selected conditional variable (mjj)
    X_cond = cond_data[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return X, X_cond, y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, cond_data, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(np.hstack([x_input,cond_data]))
    # create class labels
    y = np.zeros((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# train the generator and discriminator
def train(g_model, d_model, gan_model, train_data, cond_data, latent_dim, n_epochs=1001, n_batch=512):
    bat_per_epo = int(train_data.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    d_hist_r, d_hist_f, g_hist  = list(), list(), list()
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        print(i)
        # for j in range(bat_per_epo):
        for j in tqdm(range(bat_per_epo)):
            # get randomly selected 'real' samples
            X_real, X_c, y_real = generate_real_samples(train_data, cond_data, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(np.hstack([X_real, X_c]), y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, X_c, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(np.hstack([X_fake, X_c]), y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create new random index for conditional variables
            idx = np.random.randint(0, cond_data.shape[0], n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([X_gan, cond_data[idx]], y_gan)
            # summarize loss on this batch
            # print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                # (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            d_hist_r.append(d_loss1) 
            d_hist_f.append(d_loss2) 
            g_hist.append(g_loss)
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' %
            (i+1, d_loss1, d_loss2, g_loss))
        if i == 0 or i % 50 == 0:
            g_model.save("generator_%d.h5"%i)
            d_model.save("disciminator_%d.h5"%i)
            gan_model.save("gan_%d.h5"%i)
    # save the generator model
    #g_model.save('generator.h5')
    np.save("./disc_loss_real.npy", d_hist_r)
    np.save("./disc_loss_fake.npy", d_hist_f)
    np.save("./gan_loss.npy", g_hist)

print("************   GPU ***************")
tf.test.is_gpu_available()
gpu_devices = tf.config.list_physical_devices('GPU')
print(gpu_devices)
print("**********************************")

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# for device in gpu_devices:
#     tf.config.experimental.per_process_gpu_memory_fraction = 0.3
#     tf.config.experimental.set_memory_growth(device, True) 
#     # try:
#     #     tf.config.experimental.set_virtual_device_configuration(
#     #         gpu_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#     # except RuntimeError as e:
#     #     print(e)


datapath = "~/cGAN_dataset/"
filename_SB = datapath+"/bkg_SB_train.h5"
df_bg_sb = pd.read_hdf(filename_SB)
print(df_bg_sb.head())

train_features = [ "mj1", "mj2", "tau21j1", "tau21j2", "tau32j1", "tau32j2"]
condition_features = ["mjj"]

all_features = train_features + condition_features

print(df_bg_sb[train_features].head())


X_raw = np.array(df_bg_sb[train_features])
print("Shape of training set: ", X_raw.shape)
# np.save("./X_raw.npy", X_raw)

scaler = MinMaxScaler((-1,1)).fit(X_raw)
# scaler = StandardScaler().fit(X_raw)
X_train = scaler.transform(X_raw)

Xc_raw = np.array(df_bg_sb[condition_features])
print("Shape of conditional data: ", Xc_raw.shape)
scaler_mjj = MinMaxScaler((-1,1)).fit(Xc_raw.reshape(-1,1))
Xc_train = scaler_mjj.transform(Xc_raw.reshape(-1,1))

NOISE_DIM = 12 #64 # 128
GEN_DIM = NOISE_DIM + len(condition_features)
DISC_DIM = len(train_features) + len(condition_features)


generator = build_generator_model_dnn()
generator.summary()
print()


discriminator = build_discriminator_model_dnn()
discriminator.summary()
print()


gan_model = create_gan(discriminator,generator,NOISE_DIM)
gan_model.summary()

X_real, X_c, y_real = generate_real_samples(X_train, Xc_train, 50)
discriminator.train_on_batch(np.hstack([X_real, X_c]), y_real)

# np.save("./X_train.npy", X_train)

# ################################################################################
# #                    MNIST DataSet example 
# ################################################################################
# #
# # example of loading the fashion_mnist dataset
# from tensorflow.keras.datasets.fashion_mnist import load_data
# # load the images into memory
# (trainX, trainy), (testX, testy) = load_data()
# # summarize the shape of the dataset
# print('Train', trainX.shape, trainy.shape)
# print('Test', testX.shape, testy.shape)

# print(testy[:100])

# ################################################################################


# train model
train(generator, discriminator, gan_model, X_train, Xc_train, latent_dim=NOISE_DIM)




