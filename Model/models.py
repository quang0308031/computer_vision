import tensorflow as tf
import numpy as np
from tensorflow import keras
from .layers.module import PositionalEmbedding
from .layers.encoder_decoder import Encoder
from .layers.encoder_decoder import Decoder

@keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               dropout_rate=0.1, decay=0.9, alpha=0.9, **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.decay = decay
    self.alpha = alpha
    self.pos_embedding = PositionalEmbedding(d_model=d_model, dff=dff)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate,
                           decay=decay)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate,
                           decay=decay, alpha=alpha)

  def call(self, input):
    input = tf.cast(tf.reshape(input, shape=[-1, input.shape[-3], input.shape[-2] * input.shape[-1]]), dtype=tf.float32)
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    input = self.pos_embedding(input)

    context = self.encoder(input)  # (batch_size, context_len, d_model)

    output = self.decoder(input, context)  # (batch_size, target_len, d_model)

    return output
  
def model(num_layers=10, d_model=64, num_heads=8, dff=128, dropout_rate=0.2, decay=0.7, alpha=0.9): 
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1./255., input_shape=(20, 100, 100)),
            Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, decay=decay, alpha=alpha),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='gelu'),
            tf.keras.layers.Dense(20, activation='softmax')
        ]
    )