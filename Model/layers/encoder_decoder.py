import tensorflow as tf
import numpy as np
from tensorflow import keras
from .Attention import GlobalSelfAttention
from .module import FeedForward
from .Attention import CausalSelfAttention
from .Attention import CrossAttention
from .module import AttentionFrame

@keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*,d_model, num_heads, dff, dropout_rate=0.1, decay=0.9, **kwargs):
    super().__init__(**kwargs)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.decay = decay

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        decay=decay)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

  def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'decay': self.decay,
            'self_attention': self.self_attention,
            'ffn': self.ffn
        })
        return config
  
@keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1, decay=0.9, **kwargs):
    super().__init__(**kwargs)

    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.decay = decay

    self.enc_layers = tf.keras.Sequential([
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate,
                     decay=decay)
        for _ in range(num_layers)])
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):

    # Add dropout.
    x = self.dropout(x)

    x = self.enc_layers(x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

  def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'decay': self.decay,
            'num_layers': self.num_layers,
            'enc_layers': self.enc_layers,
            'dropout':self.dropout
        })
        return config
  
@keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1,
               decay=0.9,
               **kwargs):
    super(DecoderLayer, self).__init__(**kwargs)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.decay = decay

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        decay=decay)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        decay=decay)

    self.ffn = FeedForward(d_model, dff)

  def call(self, inputs):
    x = self.causal_self_attention(x=inputs[0])
    x = self.cross_attention(x=x, context=inputs[1])

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x, inputs[1]

  def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'decay': self.decay,
            'causal_self_attention': self.causal_self_attention,
            'cross_attention': self.cross_attention,
            'ffn': self.ffn
        })
        return config
  
@keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               dropout_rate=0.1, decay=0.9, alpha=0.9, **kwargs):
    super(Decoder, self).__init__(**kwargs)

    self.AF = AttentionFrame(alpha=alpha)

    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.decay = decay
    self.alpha = alpha

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = tf.keras.Sequential([
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate,
                     decay=decay)
        for _ in range(num_layers)])

  def call(self, x, context):
    x = self.dropout(x)
    x = self.AF(x)

    x  = self.dec_layers([x, context])

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x[0]

  def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'AF': self.AF,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'decay': self.decay,
            'alpha': self.alpha,
            'num_layers': self.num_layers,
            'dropout':self.dropout,
            'dec_layers': self.dec_layers,
        })
        return config