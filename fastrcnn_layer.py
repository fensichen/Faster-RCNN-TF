import tensorflow as tf
import numpy as np
import roi_pool



class fastrcnn_layer(object):
    """
    Crop and resize area from the feature-extracting CNN's feature map according to the ROIs generated from ROI proposal layer
    """
    def __init__(self, feature_map, roi_proposal_net):
        self.feature_map             = feature_map
        self.roi_proposal_net        = roi_proposal_net
        self.rois                    = roi_proposal_net.get_rois()
        self.im_dims                 = roi_proposal_net.im_dims
        self.num_classes             = roi_proposal_net.num_classes
        self.p_drop                  = 0.5 
        self.bn1                     = BatchNorm( name='bn1')
        self.bn2                     = BatchNorm( name='bn2')

        self.weights                 = {
            'wfc1':     tf.Variable( tf.random_normal([49*512,  1024], stddev=0.005)), 
            'wfc2':     tf.Variable( tf.random_normal([1024,1024], stddev=0.005)),
            'wfccls':   tf.Variable( tf.random_normal([1024,self.num_classes    ], stddev=0.005)),
            'wfcbbox':  tf.Variable( tf.random_normal([1024,self.num_classes*4  ], stddev=0.005))
        } 

        self.biases                  = {
            'bfc1':     tf.Variable( tf.ones([1024])),
            'bfc2':     tf.Variable( tf.ones([1024])),
            'bfccls':   tf.Variable( tf.ones([self.num_classes])),
            'bfcbbox':  tf.Variable( tf.ones([self.num_classes*4]))
        }

        self.network()
    def network(self):
        with tf.variable_scope('fast_rcnn'):
            # roi pooling
            pooled_features          = roi_pool.roi_pool(self.feature_map, self.rois, self.im_dims)            
            #pooled_features          = np.reshape( pooled_features, [ -1, 7*7*512 ] ) 
            pooled_features          = tf.contrib.layers.flatten(pooled_features)
            self.pooled_features     = pooled_features
            # Fully connected layers
            fc1                      = tf.nn.dropout( self.fc(pooled_features, self.weights['wfc1'], self.biases['bfc1']), self.p_drop)
            fc2                      = tf.nn.dropout( self.fc(fc1, self.weights['wfc2'], self.biases['bfc2']), self.p_drop)
            self.feature             = fc2

        #classifier score
        with tf.variable_scope('cls'):
            self.rcnn_cls_score      = self.fc( self.feature, self.weights['wfccls'], self.biases['bfccls'] ) 

        with tf.variable_scope('bbox'):
            self.rcnn_bbox_refine    = self.fc( self.feature, self.weights['wfcbbox'],self.biases['bfcbbox'])


    def fc(self,x,W,b):          
        h                            = tf.matmul(x, W) + b
        h                            = tf.nn.relu(h)
        return h

    def get_cls_score(self):
        return self.rcnn_cls_score

    def get_cls_prob(self):
        logits = self.get_cls_score()
        return tf.nn.softmax(logits)

    def get_bbox_refinement(self):
        return self.rcnn_bbox_refine

    def get_fast_rcnn_cls_loss(self):
        fast_rcnn_cls_score          = self.get_cls_score()
        labels                       = self.roi_proposal_net.get_labels()
        loss                         = self.fast_rcnn_cls_loss(fast_rcnn_cls_score, labels)
        return loss

    def get_fast_rcnn_bbox_loss(self):
        fast_rcc_bbox_pred           = self.get_bbox_refinement
        bbox_targets                 = self.roi_proposal_net.get_bbox_targets()
        roi_inside_weights           = self.roi_proposal_net.get_bbox_inside_weights()
        roi_outside_weights          = self.roi_proposal_net.get_bbox_outside_weights()
        loss                         = self.fast_rcc_bbox_loss(fast_rcc_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights)
        return loss

    def fast_rcnn_cls_loss(self, fast_rcnn_cls_score, labels):
        """
        corss-entropy loss 
        """
        fast_rcnn_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fast_rcnn_cls_score, lables=labels)
        fast_rcnn_cross_entropy_loss = tf.reduce_mean(fast_rcnn_cross_entropy_loss)
        return fast_rcnn_cross_entropy_loss

    def fast_rcnn_bbox_loss(self, bbox_targets, roi_inside_weights, roi_outside_weights):
        """
        calculate the fast RCNN bounding box refinement loss. Measure how well the fast RCNN is able to refine localization 

        lambda* (1/N_reg) * sum_i( pi^* x L_Reg(ti, ti^*) )
        where ti   : 4-d coordinate representing predicted bounding box
              ti^* : the ground truth bounding box associated with a positive anchor
              L_reg= R(ti - ti^*), R is the smooth L1
        """

        diff                         = tf.multiply( roi_inside_weights , fast_rcc_bbox_pred - bbox_targets )
        diff_sL1                     = self.smoothL1( diff, 1.0)
        # only count loss for posibie anchors
        roi_bbox_reg                 = tf.reduce_sum(tf.multiply(roi_outside_weights, diff_sL1), reduction_indices=[1] )
        roi_bbox_reg                 = tf.reduce_mean(roi_bbox_reg)

        roi_bbox_reg                 = 10 * roi_bbox_reg

        return roi_bbox_reg

    def smoothL1( self, x, sigma):
        conditional                  = tf.less(tf.abs(x), 1/sigma**2)
        close                        = 0.5* (sigma * 2 ) **2
        far                          = tf.abs(x) - 0.5/sigma ** 2

        return tf.where(conditional, close, far)


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name     = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)