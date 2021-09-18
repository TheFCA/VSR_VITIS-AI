"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 8th 2018
Updated Date: May 25th 2018
Author: Fernando Carrio
Email: fernando.carrio@uma.es
Changes:
 - Support for filters
 - Removal of VSR.Utils
 - Removal of bicubic_rescale (This is done now in the data pipelines)
 - Simpler and complete conv2d layer with embedded kernel initializations
SRCNN mainly for framework tests (ECCV 2014)
Ref https://arxiv.org/abs/1501.00092
"""
import tensorflow as tf
from ..Framework.SuperResolution import SuperResolution
from tensorflow.python.platform import gfile #fcarrio

class SRCNN(SuperResolution):
  """Image Super-Resolution Using Deep Convolutional Networks

  Args:
      layers: number layers to use
      filters: number of filters of conv2d(s)
      kernel: a tuple of integer, representing kernel size of each layer,
        can also be one integer to specify the same size
  """

  def conv2d_fca (self, x, filters, kernel_size, bias=False, name=""):
    seed = 789
    bi = tf.zeros_initializer()
    kr = tf.keras.regularizers.l2(self.weight_decay)
    ki = tf.keras.initializers.he_normal(seed=seed)
    nn = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1),
                    padding='same', data_format='channels_last',
                    dilation_rate=(1,1), use_bias=bias,
                    kernel_initializer=ki, kernel_regularizer=kr,
                    bias_initializer=bi, name=name) #,activation=tf.nn.relu
    nn.build(x.shape.as_list())
    x = nn(x)
    return x  

  def __init__(self, layers=3, filters=(64,32,1), kernel=(9, 5, 5),
               name='srcnn', **kwargs):
    super(SRCNN, self).__init__(**kwargs)
    self.name = name

    self.layers = layers
    # f.carrio adding support for different filter layers
    self.filters = (filters)
    self.kernel_size = (kernel)
    

  def conv2d (self,x, filters, kernel_size, name, bias=False):
      seed = 789
      weight_decay = 0.0001
      bi = tf.zeros_initializer()
      kr = tf.keras.regularizers.l2(weight_decay)
      ki = tf.keras.initializers.he_normal(seed=seed)

      nn = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1),
                      padding='same', data_format='channels_last',
                      dilation_rate=(1,1), use_bias=bias,
                      kernel_initializer=ki, kernel_regularizer=kr,
                      bias_initializer=bi,name=name) #,activation=tf.nn.relu

      nn.build(x.shape.as_list())
      x = nn(x)
      return x

  def build_graph(self):
    super(SRCNN, self).build_graph()
    x = self.inputs[-1]
    with tf.variable_scope(self.name):
        x = self.conv2d(x, self.filters[0], self.kernel_size[0], name="InputLayer", bias=True)
        x = tf.nn.relu(x)     
        for i in range(1, self.layers - 1):
            x = self.conv2d(x, self.filters[i], self.kernel_size[i], name=("Layer" + str(i)), bias=True)  
            x = tf.nn.relu(x)      
            x = self.conv2d(x, 1, self.kernel_size[-1], name="OutputLayer", bias=True)
        self.outputs.append(x)


  def build_loss(self):
    with tf.name_scope('loss'):
      y_pred = self.outputs[-1]
      y_true = self.label[-1]
      mse = tf.losses.mean_squared_error(y_true, y_pred)
      loss = tf.losses.get_total_loss()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.loss.append(opt.minimize(loss, self.global_steps))
      self.train_metric['loss'] = loss
      self.metrics['mse'] = mse
      self.metrics['psnr'] = tf.reduce_mean(tf.image.psnr(self.label[-1], self.outputs[-1], max_val=255))
      self.metrics['ssim'] = tf.reduce_mean(tf.image.ssim(self.label[-1], self.outputs[-1], max_val=255))


##################
## F. Carrió
## This function permits the execution an simulation of pre-trained frozen graphs in pb format.
## 
##################

  def compile_frozen_graph(self):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      
      with gfile.FastGFile(self.pre_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Variable Initialization
        self.global_steps = tf.Variable(0, trainable=False, name='global_step')
        self.training_phase = tf.placeholder_with_default(False, shape=None, name='train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # PlaceHolders Initialization        
        self.label.append(
        tf.placeholder(tf.float32, shape=[ None, None, None, self.channel], name='label/hr')) #None
        
        self.inputs.append( #uint8
          tf.placeholder(tf.float32, shape=[ None,None, None, self.channel],name='input/lr'))    
        # Definition of the graph input
        g_in = tf.import_graph_def(graph_def, input_map={'input/lr': self.inputs[-1]})
   
      # We define the output of the tensor we want to monitor. This can be useful to analyze intermediate signals and tensors
      self.tensor_out = tf.get_default_graph().get_tensor_by_name('import/'+self.output_node+':0')
      Result = self.outputs.append(self.tensor_out)
      self.summary_graph()
  
  def compile_quantized_graph(self):
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)

      with gfile.FastGFile(self.pre_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Variable Initialization
        self.global_steps = tf.Variable(0, trainable=False, name='global_step')
        self.training_phase = tf.placeholder_with_default(False, shape=None, name='train')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # PlaceHolders Initialization        
        self.label.append(
        tf.placeholder(tf.float32, shape=[ None, None, None, self.channel], name='label/hr')) #None

        self.inputs.append( #uint8
          tf.placeholder(tf.float32, shape=[ None,None, None, self.channel],name='input/lr'))
        # Definition of the graph input
        g_in = tf.import_graph_def(graph_def, input_map={'input/lr': self.inputs[-1]})

      # We define the output of the tensor we want to monitor. This can be useful to analyze intermediate signals and tensors
      self.tensor_out = tf.get_default_graph().get_tensor_by_name('import/'+self.output_node+':0')
      Result = self.outputs.append(self.tensor_out)
      self.summary_quantized_graph()

  def summary_quantized_graph(self):
    for t in tf.get_default_graph().get_operations():
      if ('aquant') in t.name.split('/')[-1]:
        self.tensor_scales[str(t.name+':0')] = t.get_attr('quantize_pos')
        self.tensors_order.append(t.name+':0')
        if t.name.split('/')[-2]!='lr':
          self.tensors[t.name+':0'] = tf.get_default_graph().get_tensor_by_name(t.name+':0')
          tt = tf.get_default_graph().get_tensor_by_name(t.name+':0')
          self.tensors['/'.join(t.name.split('/')[:-1])+':0']=tf.get_default_graph().get_tensor_by_name(str('/'.join(t.name.split('/')[:-1])+':0'))
        else:
          self.tensors['import/input/lr:0'] = tf.get_default_graph().get_tensor_by_name('input/lr:0')
          self.tensors['import/input/lr/aquant:0'] = tf.get_default_graph().get_tensor_by_name('import/input/lr/aquant:0')

      elif ('wquant') in (t.name.split('/')[-1]).split('_')[0]:
        self.tensor_scales[t.name+':0'] = t.get_attr('quantize_pos')
        self.tensors[t.name+':0'] = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors['/'.join(t.name.split('/')[:-1])+':0']=tf.get_default_graph().get_tensor_by_name(str('/'.join(t.name.split('/')[:-1])+':0'))
  
  def summary_graph(self):

    for t in tf.get_default_graph().get_operations():
      if ('Conv2D') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('BiasAdd') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('Relu') in (t.name.split('/')[-1]).split('_')[0]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('add') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('FusedBatchNormV3') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('kernel') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('bias') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('gamma') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('beta') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('moving_mean') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('moving_variance') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name(t.name+':0')
        self.tensors[t.name+':0'] = tensor_aux
      elif ('lr') in t.name.split('/')[-1]:
        tensor_aux = tf.get_default_graph().get_tensor_by_name('input/lr:0')
        self.tensors[t.name+':0'] = tensor_aux
    self.tensors['input/lr:0'] = tf.get_default_graph().get_tensor_by_name('input/lr:0')
