#convnet.py
import tensorflow as tf
import numpy as np
import scipy as scp
import scipy.misc 

class ConvNetVgg16(object):

    def __init__(self, vgg16_model_path):
        self.model = np.load(vgg16_model_path, encoding="latin1").item()

    def get_weight(self, name):
        with tf.variable_scope(name) as scope:
            w_tensor     = self.model[name][0]
            init_        = tf.constant_initializer( value = w_tensor , dtype = tf.float32)
            shape_       = w_tensor.shape
            return tf.get_variable(name, initializer = init_, shape = shape_ )    

    def get_bias(self, name):
        with tf.variable_scope(name) as scope:
            b_tensor     = self.model[name][1]
            init_        = tf.constant_initializer( value = b_tensor, dtype = tf.float32)
            shape_       = b_tensor.shape
            return tf.get_variable(name + "Bias", initializer = init_, shape = shape_)

    def conv2d(self, x, name):
        with tf.variable_scope(name) as scope:
            W            = self.get_weight(name)
            b            = self.get_bias(name)
            x            = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
            x            = tf.nn.bias_add(x,b)
            return tf.nn.relu(x)

    def max_pool(self, x, name):        
        return tf.nn.max_pool( x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def inference(self, img):
       
        self.conv1_1     = self.conv2d(img,            "conv1_1")      
        self.conv1_2     = self.conv2d(self.conv1_1,   "conv1_2")
        self.pool1       = self.max_pool(self.conv1_2, "pool1"  )

        self.conv2_1     = self.conv2d(self.pool1,     "conv2_1")
        self.conv2_2     = self.conv2d(self.conv2_1,   "conv2_2")
        self.pool2       = self.max_pool(self.conv2_2, "pool2"  )
        
        self.conv3_1     = self.conv2d(self.pool2,     "conv3_1")
        self.conv3_2     = self.conv2d(self.conv3_1,   "conv3_2")    
        self.conv3_3     = self.conv2d(self.conv3_2,   "conv3_3")
        self.pool3       = self.max_pool(self.conv3_3, "pool3"  )
        
        self.conv4_1     = self.conv2d(self.pool3,     "conv4_1")     
        self.conv4_2     = self.conv2d(self.conv4_1,   "conv4_2")
        self.conv4_3     = self.conv2d(self.conv4_2,   "conv4_3")
        self.pool4       = self.max_pool(self.conv4_3, "pool4"  )

        self.conv5_1     = self.conv2d(self.pool4,     "conv5_1")      
        self.conv5_2     = self.conv2d(self.conv5_1,   "conv5_2")
        self.conv5_3     = self.conv2d(self.conv5_2,   "conv5_3")
        #self.pool5       = self.max_pool(self.conv5_3, "pool5"  )
        #self.output      = self.pool5       
        self.output      = self.conv5_3
        return self.output

    def get_features(self):
        return self.output

if __name__ == '__main__':

    vgg16 = ConvNetVgg16( '/home/fensi/nas/vgg16/vgg16.npy')
    img   = scipy.misc.imread('/home/fensi/nas/demo/tabby_cat.png')
    img   = img.astype( np.float32 )
    img   = np.expand_dims( img, axis = 0)
    vgg16.inference(img) 
