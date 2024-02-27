import tensorflow as tf
import numpy as np
from tensorflow import keras
from module import EMA_Layer
import math

@keras.saving.register_keras_serializable()
class MultiHeadAttention_OVR(tf.keras.layers.MultiHeadAttention):
  def __init__(self, num_heads: int, key_dim: int, decay: float, **kwargs):
    super().__init__(num_heads, key_dim, **kwargs)
    self.EMA = EMA_Layer(decay)


  def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self.EMA(attention_scores)
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

  def get_config(self):
        config = super(MultiHeadAttention_OVR, self).get_config()
        config.update({
            'EMA': self.EMA
        })
        return config
  
@keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention_OVR(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def get_config(self):
        config = super(BaseAttention, self).get_config()
        config.update({
            'mha': self.mha,
            'layernorm': self.layernorm,
            'add':self.add
        })
        return config
  
@keras.saving.register_keras_serializable()
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
@keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
@keras.saving.register_keras_serializable()
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

