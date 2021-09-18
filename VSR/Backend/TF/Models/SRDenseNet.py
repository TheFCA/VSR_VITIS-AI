"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 19th 2018
Modified by: Fernando Carrió
Added: metrics monitoring and SGD support - Model adjustments

Image Super-Resolution Using Dense Skip Connections
See http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
"""

import tensorflow as tf

from ..Arch import Dense
from ..Framework.SuperResolution import SuperResolution


def _denormalize(inputs):
  return (inputs + 0) * 255


def _normalize(inputs):
  return inputs / 255


class SRDenseNet(SuperResolution):
  """Image Super-Resolution Using Dense Skip Connections.
  Args:
      n_blocks: number of dense blocks.
  """

  def __init__(self, name='srdensenet', n_blocks=8, **kwargs):
    super(SRDenseNet, self).__init__(**kwargs)
    self.name = name
    self.n_blocks = n_blocks

  def build_graph(self):
    super(SRDenseNet, self).build_graph()
    with tf.variable_scope(self.name):
      inputs_norm = _normalize(self.inputs[-1])
      feat = [self.conv2d(inputs_norm, 64, 3)]
      for i in range(self.n_blocks):
        feat.append(Dense.dense_block(self, feat[-1]))
      bottleneck = self.conv2d(tf.concat(feat, -1), 256, 1)
      sr = self.upscale(bottleneck, 'deconv', direct_output=False)
      sr = self.conv2d(sr, self.channel, 3)
      self.outputs.append(_denormalize(sr))

  def build_summary(self):
    super(SRDenseNet, self).build_summary()
    tf.summary.image('sr', self.outputs[-1])

  def build_loss(self):
    with tf.name_scope('loss'):
      y_true = self.label[-1] # entire dataset
      y_pred = self.outputs[-1] # entire outputs

      mse = tf.losses.mean_squared_error(y_true, y_pred)
      loss = tf.add_n([mse] + tf.losses.get_regularization_losses())
      optimizer = 'Adam'
      if optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(self.learning_rate)
        mixed_precision = False
        if mixed_precision:
          opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)        
      elif optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        if self.grad_clip > 0:
          grads_and_vars = []
          for grad, var in opt.compute_gradients(loss):
            grads_and_vars.append((
              tf.clip_by_value(
                  grad,
                  -self.grad_clip / self.learning_rate,
                  self.grad_clip / self.learning_rate),
              var))
          op = opt.apply_gradients(grads_and_vars, self.global_steps)
        else:
          op = opt.minimize(loss, self.global_steps)        
      self.loss.append(op)

      self.train_metric['loss'] = loss
      self.metrics['regloss'] =  tf.cast(sum(tf.losses.get_regularization_losses()),'float32') # fcarrio
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(
          tf.image.psnr(y_true, y_pred, 255))
      self.metrics['ssim'] = tf.reduce_mean(
          tf.image.ssim(y_true, y_pred, 255))