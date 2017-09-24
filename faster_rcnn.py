#faster_rcnn.py
import tensorflow as tf
import numpy as np
import scipy as scp
import scipy.misc

import region_proposal_network
import convnet
import data_handler
import roi_proposal_layer
import fastrcnn_layer

HEIGHT = 375
WIDTH  = 1242

class FasterRCNN(object):
    def __init__(self):
        # initial placeholder dictionary
        self.X                   = {}
        self.gt_bbox             = {}
        self.im_dims             = {}

        # for training
        self.X['train']          = tf.placeholder(tf.float32, [None, None, None, 3]) # [ batch_size, height, width, channel]
        self.gt_bbox['train']    = tf.placeholder(tf.int32,   [None, 5])
        self.im_dims['train']    = tf.placeholder(tf.int32,   [None, 2])

        # for testing
        self.X['test']           = tf.placeholder(tf.float32, [None, None, None, 3])
        self.im_dims['test']     = tf.placeholder(tf.int32,   [None, 2])
        """
        Define the network output 
        """
        self.cnn                 = {}
        self.rpn                 = {}
        self.roi                 = {}
        self.fastrcnn            = {}

        self.anchor_scale        = [ 8, 16, 32 ] # Anchor boxes will have dimensions scales * 16 * ratio in the image space
        self.num_steps           = 100
        self.datahandler         = data_handler.DataHandler()
        self.batch_size          = 1

    def read_data(self, train_filename, label_filename = None):
        
        self.datahandler.get_file_list( train_filename, label_filename)
  
    def network(self):

        # Training network
        with tf.variable_scope('model') as scope:
            self.faster_rcnn(self.X['train'], self.gt_bbox['train'], Mode = 'train')

        # Inference network
        with tf.variable_scope('model', reuse=True):
            self.faster_rcnn(self.X['test'], None, Mode = 'test')


    def faster_rcnn(self, x, gt_bbox, Mode = 'train' ):
  
        vgg16                    = convnet.ConvNetVgg16('/home/fensi/nas/vgg16/vgg16.npy')
        self.cnn[ Mode ]         = vgg16.inference(x) 
        features                 = vgg16.get_features()

        #run Region Proposal Network
        self.rpn[ Mode ]         = region_proposal_network.RegionProposalNetwork(features, self.gt_bbox['train'], self.im_dims['train'], self.anchor_scale, Mode)
        #run ROI Pooling
        self.roi[ Mode ]         = roi_proposal_layer.roi_proposal_layer(self.rpn[Mode], self.gt_bbox['train'], self.im_dims['train'], Mode)            
       	#run R-CNN Classification
       	self.fastrcnn[ Mode ]    = fastrcnn_layer.fastrcnn_layer(features, self.roi[Mode])

    def optimizer(self):
        """ Define loss and initiali optimizer """
        self.init_feature        = self.cnn ['train']
        self.rpn_cls_loss        = self.rpn['train'].get_rpn_cls_loss()
        self.rpn_input_feat      = self.rpn['train'].get_rpn_input_feature()
        #self.rpn_bbox_loss       = self.rpn['train'].get_rpn_bbox_loss()
        #self.fast_rcnn_cls_loss  = self.fastrcnn['train'].get_fast_rcnn_cls_loss()
        #self.fast_rcnn_bbox_loss = self.fastrcnn['train'].get_fast_rcnn_bbox_loss()
        #self.cost                = tf.reduce_sum( self.rpn_cls_loss + self.rpn_bbox_loss + self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss )
        self.cost                = self.rpn['train'].get_rpn_cls_score() 
        self.rpn_label           = self.rpn['train'].get_rpn_labels()
        self.rpn_bbox_pred       = self.rpn['train'].get_rpn_bbox_pred()
        self.roi_blob            = self.roi['train'].get_rois()

        
        self.pooled_features     = self.fastrcnn['train'].pooled_features

        self.optimize_op         = tf.train.AdamOptimizer(1e-3).minimize(self.cost)



    def train(self):

        tf_feed                       = ( self.X['train'], self.gt_bbox['train'], self.im_dims['train'] )
        self.network()
        self.optimizer() 
        with tf.Session() as sess:
            init                      = tf.global_variables_initializer()
            sess.run(init)
            for step in range(0, self.num_steps):
                self.datahandler.shuffle()
                for start in range(0, self.datahandler.num() - self.datahandler.num() % self.batch_size, self.batch_size):
                    end               = start + self.batch_size
                    data,label,im_dim = self.datahandler.load_data( start, end, is_training = True)
                    #_, loss,loss_rpn_cls,loss_rpn_bbox, loss_fast_rcnn_cls, loss_fast_rcnn_bbox = sess.run( [ self.optimizer, self.cost, self.rpn_cls_loss, self.rpn_bbox_loss, self.fast_rcnn_cls_loss, self.fast_rcnn_bbox_loss ], feed_dict = {tf_feed[0]: data, tf_feed[1]: label, tf_feed[2]: im_dim} )
                    #self.optimizer()
                    #sess.run( self.optimize_op, feed_dict = {self.X['train']: data, self.gt_bbox['train']: label, self.im_dims['train']: im_dim} )
                    rpn_input_feat, rpn_cls_score, rpn_label, rpn_cls_loss, rpn_bbox_pred, roi_blob, pooled_features = \
                    sess.run( [self.init_feature, self.cost, self.rpn_label, self.rpn_cls_loss, self.rpn_bbox_pred, self.roi_blob, self.pooled_features]
                    ,feed_dict = {self.X['train']: data, self.gt_bbox['train']: label, self.im_dims['train']: im_dim} )
                    print "rpn_input_feat.shape", rpn_input_feat.shape
                    print "rpn_cls_score", rpn_cls_score.shape
                    print "rpn_bbox_pred", rpn_bbox_pred.shape
                    print "pooled_features.shape", pooled_features.shape
                    #print "rpn_label", rpn_label.shape, rpn_label
                    #print "rpn_cls_loss", rpn_cls_loss
                    #print "roi_blob", roi_blob.shape, roi_blob
                    exit(0)


def main():
    detector        = FasterRCNN()  
    detector.read_data('/home/fensi/nas/KITTI_OBJECT/train.txt', '/home/fensi/nas/KITTI_OBJECT/label.txt')
    detector.train()





if __name__ == '__main__':
    main()
    #vgg16          = convnet.ConvNetVgg16('/home/fensi/nas/vgg16/vgg16.npy')
    #img            = scipy.misc.imread('/home/fensi/nas/demo/tabby_cat.png')
    #img            = np.float32(img)
    #img            = np.expand_dims(img, axis=0)
    #output          = vgg16.inference(img)

    



