'''
Author:- Abhishek Kuriyal

Self made CNN model without use of Python's prebuilt Deep learning frameworks.
This model only uses Numpy for matrix operations hence have very less dependencies.

Due to the complexity of CNN, this model fails to predict output for multi-colored channel
images, however works perfectly for grayscale images like mnist dataset.
For sake of complexity, we started to program this CNN considering only grayscale images
However extending the model to multi color channels is more complex than presumed
and requires complete restucturing of the code.

Since colors provide important features for any gammatone based spectrogram, we refrained 
to mentioned the results of this script in Execution video.

'''



import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from keras.datasets import mnist


class ConvOp:
   def __init__(self, num_filters, filter_size):
     self.num_filters = num_filters
     self.filter_size = filter_size
     # Generating random filters and normalizing by dividing with filter_size
     self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size * filter_size)    

   def image_region(self, image):
    self.image = image
    height, width = self.image.shape
    for j in range(height - self.filter_size + 1):      # dimension of filter - n - f + 1, no padding involved
      for k in range(width - self.filter_size + 1):
        image_patch = image[j: (j + self.filter_size), k:(k + self.filter_size)]
        yield image_patch, j, k

   def forward_prop(self, image):
    height, width = image.shape
    conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
    for image_patch, i, j in self.image_region(image):
        conv_out[i, j] = np.sum(image_patch*self.conv_filter, axis = (1,2))
    return conv_out

   def back_prop(self, dL_dout, learning_rate):
    dL_dF_params = np.zeros(self.conv_filter.shape)
    for image_patch, i, j in self.image_region(self.image):
      for k in range(self.num_filters):
        dL_dF_params[k] += image_patch*dL_dout[i, j, k]

    self.conv_filter -= learning_rate*dL_dF_params
    return dL_dF_params
    
    
    
    
class MaxPool:      # No weights and Biases involved in MaxPool, hence no learning part, in Conv_layer, random filters were generated.
  def __init__(self, filter_size):
    self.filter_size = filter_size

  def image_region(self, image):
    self.image = image

    new_height = self.image.shape[0] // self.filter_size
    new_width = self.image.shape[1] // self.filter_size

    for i in range(new_height):
      for j in range(new_width):
        image_patch = image[(i*self.filter_size):(i*self.filter_size + self.filter_size), (j*self.filter_size): (j*self.filter_size + self.filter_size)]
        yield image_patch, i, j

  def forward_prop(self, image):
    height, width, num_filters = image.shape
    output = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))

    for image_patch, i, j in self.image_region(image):
      output[i, j] = np.amax(image_patch, axis=(0,1))


    return output


  def back_prop(self, dL_dout):
    dL_dmax_pool = np.zeros(self.image.shape)
    for image_patch, i, j in self.image_region(self.image):
      height, width, num_filters = image_patch.shape
      maximum_val = np.amax(image_patch, axis=(0,1))

      for i1 in range(height):
        for j1 in range(width):
          for k1 in range(num_filters):
            if image_patch[i1, j1, k1] == maximum_val[k1]:
              dL_dmax_pool[i*self.filter_size + i1, j*self.filter_size + j1, k1] = dL_dout[i,j,k1]

      return dL_dmax_pool
      
      
      
class SoftMax:
  def __init__(self, input_shape, softmax_shape):
    self.softmax_shape = softmax_shape
    self.weight = np.random.randn(input_shape, self.softmax_shape)/input_shape
    self.bias = np.zeros(self.softmax_shape)
    

  def forward_prop(self, image):
    self.image = image
    self.orig_im_shape = self.image.shape
    image_modified = self.image.flatten()
    self.modified_input = image_modified
    output_val = np.dot(image_modified, self.weight)  + self.bias
    self.out = output_val
    exp_out = np.exp(output_val)
    return exp_out/np.sum(exp_out, axis=0)

  def back_prop(self, dL_dout, learning_rate):
    for i, grad in enumerate(dL_dout):
      if grad == 0:
        continue

      transformation_eq = np.exp(self.out)

      S_total = np.sum(transformation_eq)

      # Gradients with respect to out(z)

      dy_dz = -transformation_eq[i]*transformation_eq/ (S_total **2)
      dy_dz[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

      # Gradients of totals against weights/biases/input
      dz_dw = self.modified_input
      dz_db = 1
      dz_d_inp = self.weight

      # Gradients of loss  against totals

      dL_dz = grad * dy_dz

      # Gradients of loss against weights/biases/input
      dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]     # @ stands for Matrix multiplication
      dL_db = dL_dz * dz_db
      dL_d_inp = dz_d_inp @ dL_dz

      self.weight -= learning_rate*dL_dw
      self.bias -= learning_rate*dL_db

      return dL_d_inp.reshape(self.orig_im_shape)
      

(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_images = X_train[:1500]
train_labels = y_train[:1500]
test_images = X_test[:1500]
test_labels = y_test[:1500]


conv = ConvOp(40,3)     # 28 x 28 x 1 -> 26 x 26 x 40
pool = MaxPool(2)      # 26 x 26 x 40 -> 13 x 13 x 40
softmax = SoftMax(13*13*40, 10)    # 13 x 13 x 40 -> 13*13*40 x 1 -> 10      
      


 

def cnn_forward_prop(image, label):
  out_p = conv.forward_prop((image/255) - 0.5)
  out_p = pool.forward_prop(out_p)
  out_p = softmax.forward_prop(out_p)

  # Calculate cross-entropy loss and accuracy
  cross_ent_loss = -np.log(out_p[label])
  accuracy_eval = 1 if np.argmax(out_p) == label else 0

  return out_p, cross_ent_loss, accuracy_eval
  
 
 

def training_cnn(image, label, learning_rate=0.01):
    # Forward
    out, loss, acc = cnn_forward_prop(image, label)

    # Calculating initial gradient

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    grad_back = softmax.back_prop(gradient, learning_rate)
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learning_rate)

    return loss, acc

      

for epoch1 in range(4):
  print("Epoch %d --->" % (epoch1 + 1))

  shuffle_data = np.random.permutation(len(train_images))
  train_images = train_images[shuffle_data]
  train_labels = train_labels[shuffle_data]

  loss = 0
  num_correct = 0

  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i%100 == 0:
      print(" %d steps out of 100 steps: Average Loss %.3f and Accuracy: %d"%(i+1, loss / 100, num_correct))
      loss = 0
      num_correct = 0

    ll, accu = training_cnn(im, label)
    loss += ll
    num_correct += accu
    
    
    
