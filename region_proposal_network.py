# region_proposal_network.py
import tensorflow as tf
import numpy as np
import anchor_target_layer 

class RegionProposalNetwork(object):
    """
    From convolutional feature map, generate bounding box, relative to anchor point, and give an objectness score to each candidate

    """

    def __init__(self, feature_vector, ground_truth, im_dims, anchor_scale, Mode):
        self.feature_vector     = feature_vector
        self.ground_truth       = ground_truth
        self.im_dims            = im_dims
        self.anchor_scale       = anchor_scale

        self.RPN_OUTPUT_CHANNEL = 512
        self.RPN_KERNEL_SIZE    = 3
        self.feat_stride        = 16

        self.weights            = {
        'w_rpn_conv1'     : tf.Variable(tf.random_normal([ self.RPN_KERNEL_SIZE, self.RPN_KERNEL_SIZE, 512, self.RPN_OUTPUT_CHANNEL ], stddev = 0.01)),
        'w_rpn_cls_score' : tf.Variable(tf.random_normal([ 1, 1, self.RPN_OUTPUT_CHANNEL, 18  ], stddev = 0.01)),
        'w_rpn_bbox_pred' : tf.Variable(tf.random_normal([ 1, 1, self.RPN_OUTPUT_CHANNEL, 36  ], stddev = 0.01))
        }

        self.mode               = Mode # train or test

        self.build_rpn()       
  
  
    def build_rpn(self):

        # rpn_conv1
        # slide a network on the feature map, for each nxn (n = 3), use a conv kernel to produce another feature map.
        # each pixel in this fature map in an anchor 
        ksize      = self.RPN_KERNEL_SIZE
        feat       = tf.nn.conv2d( self.feature_vector, self.weights['w_rpn_conv1'], strides = [1, 1, 1, 1], padding = 'SAME' )
        feat       = tf.nn.relu( feat )
        self.feat  = feat

        # for each anchor, propose k anchor boxes, 
        # for each box, regress: objectness score and coordinates

        # box-classification layer ( objectness scor)
        with tf.variable_scope('cls'):
            self.rpn_cls_score = tf.nn.conv2d(feat, self.weights['w_rpn_cls_score'], strides = [ 1, 1, 1, 1], padding = 'SAME')

        # bounding-box prediction 
        with tf.variable_scope('reg'): 
            self.rpn_reg_pred  = tf.nn.conv2d(feat, self.weights['w_rpn_bbox_pred'], strides = [1, 1, 1, 1], padding = 'SAME')
       
        # Anchor Target Layer ( anchor and delta )
        with tf.variable_scope('anchor'):
            if self.mode == 'train':
                self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                anchor_target_layer.anchor_target_layer( self.rpn_cls_score, self.ground_truth, self.im_dims, self.feat_stride, self.anchor_scale )
                
    def get_rpn_input_feature(self):
        return self.feat

    def get_rpn_cls_score(self):
        return self.rpn_cls_score

    def get_rpn_labels(self):
        return self.rpn_labels

    def get_rpn_bbox_pred(self):
        return self.rpn_reg_pred

    def get_rpn_bbox_targets(self):
        return self.rpn_bbox_targets

    def get_rpn_bbox_inside_weights(self):
        return self.rpn_bbox_inside_weights

    def get_rpn_bbox_outside_weights(self):
        return self.rpn_bbox_outside_weights

    def get_rpn_bbox_loss(self):
        rpn_bbox_pred            = self.get_rpn_bbox_pred()
        rpn_bbox_targets         = self.get_rpn_bbox_targets()
        rpn_bbox_inside_weights  = self.get_rpn_bbox_inside_weights()
        rpn_bbox_outside_weights = self.get_rpn_bbox_outside_weights()  
        return self.rpn_bbox_loss( rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

    def get_rpn_cls_loss(self):
        rpn_cls_score            = self.get_rpn_cls_score()
        rpn_labels               = self.get_rpn_labels()
        return self.rpn_cls_loss(rpn_cls_score, rpn_labels)

    def rpn_cls_loss(self, rpn_cls_score, rpn_labels):


        #self.batch_size          = 1 
        #self.num_classes         = 2
        #logits                   = tf.reshape( rpn_cls_score, [self.batch_size, -1, self.num_classes] ) 
        #labels                   = tf.reshape( rpn_labels, [self.batch_size , -1]) 
        #self.rpn_cls_score = logits
        #self.rpn_labels    = labels
        #rpn_cross_entropy        = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels= labels)
        #rpn_cross_entropy        = tf.reduce_mean(rpn_cross_entropy)
        #return rpn_cross_entropy     

        shape                     = tf.shape(rpn_cls_score)
        
        # Stack all classification scores into 2D matrix
        rpn_cls_score             = tf.transpose(rpn_cls_score,[0,3,1,2])
        rpn_cls_score             = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
        rpn_cls_score             = tf.transpose(rpn_cls_score,[0,2,3,1])
        rpn_cls_score             = tf.reshape(rpn_cls_score,[-1,2])
        
        # Stack labels
        rpn_labels                = tf.reshape(rpn_labels,[-1])
        
        # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
        rpn_cls_score             = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
        rpn_labels                = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
        # Cross entropy error
        rpn_cross_entropy         = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))
    
        return rpn_cross_entropy


    def rpn_bbox_loss(self, rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
        
        rpn_bbox_targets          = tf.transpose( rpn_bbox_targets,   [ 0, 2, 3, 1])
        rpn_inside_weights        = tf.transpose( rpn_inside_weights, [ 0, 2, 3, 1])
        rpn_outside_weights       = tf.transpose( rpn_outside_weights,[ 0, 2, 3, 1]) 
        
        diff                      = tf.multiply( rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
        diff_sL1                  = smoothL1(diff,3.0)

        rpn_bbox_reg              = 10*tf.reduce_sum(tf.multiply(tf.rpn_outside_weights, diff_sL1))
        return rpn_bbox_reg

    def smoothL1( self, x, sigma):
        conditional               = tf.less(tf.abs(x), 1/sigma**2)
        close                     = 0.5* (sigma * 2 ) **2
        far                       = tf.abs(x) - 0.5/sigma ** 2

        return tf.where(conditional, close, far)