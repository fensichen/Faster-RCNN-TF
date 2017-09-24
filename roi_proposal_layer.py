import tensorflow as tf
import numpy as np
import proposal_layer
import proposal_target_layer
import rpn_softmax


class roi_proposal_layer(object):
	"""
	propose highest scoring boxes to the RCNN classifer
	"""
	def __init__(self, rpn_net, gt_boxes, im_dims, mode):
		self.rpn_net       = rpn_net
		self.gt_boxes      = gt_boxes
		self.im_dims       = im_dims
		self.mode          = mode

		self.rpn_cls_score = rpn_net.get_rpn_cls_score()
		self.rpn_bbox_pred = rpn_net.get_rpn_bbox_pred() 
		
		self.num_classes   = 8; 
		self.anchor_scales = [ 8, 16, 32 ];
		
		self.network()

	def network(self):
		
		with tf.variable_scope('roi_proposal'):
			# convert score to probablility
			self.rpn_cls_prob = rpn_softmax.rpn_softmax(self.rpn_cls_score)
			# determine the best proposal
			self.blobs        = proposal_layer.proposal_layer( rpn_bbox_cls_prob = self.rpn_cls_prob, rpn_bbox_pred = self.rpn_bbox_pred, mode  = self.mode, 
				im_dims = self.im_dims, feat_strides = self.rpn_net.feat_stride, anchor_scales = self.anchor_scales)

			if self.mode == 'train':
				self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
				proposal_target_layer.proposal_target_layer( rpn_rois = self.blobs, gt_boxes = self.gt_boxes, num_classes = self.num_classes)
				

	def get_rois(self):
		if self.mode == 'train':
			return self.rois
		else:
			return self.blobs


	def get_labels(self):
		return self.labels

	def get_bbox_targets(self):
		return self.bbox_targets

	def get_bbox_inside_weights(self):
		return self.bbox_inside_weights

	def get_bbox_outside_weights(self):
		return self.bbox_outside_weights