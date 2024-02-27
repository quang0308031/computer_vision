import tensorflow as tf
import numpy as np
from tensorflow import keras

@keras.saving.register_keras_serializable()
class EMA_Layer(tf.keras.layers.Layer):
    def __init__(self, alpha=0.9, **kwargs):
        super(EMA_Layer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        split = tf.split(x, x.shape[-2], axis=-2)
        ema_tensor = split[0]
        for i in range(1, x.shape[-2]):
            ema_tensor = tf.concat([ema_tensor, split[i] * self.alpha + ema_tensor[... , -1:, :] * (1 - self.alpha)], axis=-2)

        return ema_tensor

    def get_config(self):
        config = super(EMA_Layer, self).get_config()
        config.update({
            'alpha': self.alpha
        })
        return config
    
@keras.saving.register_keras_serializable()
class AttentionFrame(tf.keras.layers.Layer):
  def __init__(self, alpha, **kwargs) -> None:
    super().__init__(**kwargs)
    self.alpha = alpha
    self.EMA1 = EMA_Layer(alpha = alpha)
    self.EMA2 = EMA_Layer(alpha = alpha*1.5)

  def build(self, input_shape):
    self.w_attn = tf.keras.layers.EinsumDense('...b,bc->...c', output_shape=[input_shape[1]], activation='sigmoid', bias_axes='c')
    self.idx_attn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(int(input_shape[1]), activation='tanh'),
            tf.keras.layers.Dense(int(input_shape[1]), activation='relu'),
            tf.keras.layers.Dense(input_shape[1], activation='linear'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.LayerNormalization()
        ]
    )
    super().build(input_shape)

  def call(self, input):
    output = self.EMA1(input)
    output += self.EMA2(output)
    output = self.idx_attn(output)
    output = self.w_attn(output)
    output = output[..., tf.newaxis] * input
    return output

  def get_config(self):
        config = super(AttentionFrame, self).get_config()
        config.update({
            'alpha': self.alpha,
            'EMA1': self.EMA1,
            'EMA2': self.EMA2,
            'w_attn': self.w_attn,
            'idx_attn':self.idx_attn
        })
        return config
  
@keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, **kwargs) -> None:
    super().__init__(**kwargs)
    self.d_model = d_model
    self.dff = dff

    self.LN = tf.keras.layers.LayerNormalization()

  def build(self, input_shape):
    self.cvrt = tf.keras.layers.EinsumDense('...b,bc->...c', output_shape=[self.d_model], activation='relu', bias_axes='c')
    self.position = self.add_weight(name="position", shape=(input_shape[1:]),
                              initializer=tf.initializers.Constant(tf.stack([tf.range(1., input_shape[1] + 1)] * input_shape[-1])),
                              trainable=False)
    self.MLP_pos = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(self.d_model, activation="tanh"),
            tf.keras.layers.Dense(self.dff, activation="relu"),
            tf.keras.layers.Dense(self.d_model, activation="sigmoid")
        ]
    )
    super().build(input_shape)

  def call(self, input):
    position = self.MLP_pos(self.position)
    output = self.cvrt(input)
    output = tf.add(output, position)
    output = self.LN(output)
    return output

  def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'LN': self.LN,
            'cvrt':self.cvrt,
            'position': self.position.numpy().tolist(),
            'MLP_pos':self.MLP_pos
        })
        return config
  
@keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

  def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'seq': self.seq,
            'add': self.add,
            'layer_norm': self.layer_norm
        })
        return config