import tensorflow as tf
import numpy as np


def rpn_softmax(rpn_cls_score):
	"""
	Resahpe the rpn_cls_score (n, W, H, 2k) to take softmax. Convert scores to probablilities
	ex. 9 anchors, n sample minibatch, convolutional features map of dimentions W*H 
	rpn_cls_score: (n, W, H, 18)
	return rpn_cls_prob
	"""
	with tf.variable_scope('rpn_softmax'):
		# input shape dimension
		shape        = tf.shape(rpn_cls_score)

		a            = tf.reshape(rpn_cls_score, (shape[0], shape[1], shape[2], 2, -1) )  
		a  			 = tf.transpose(a, [0,1,2,4,3]) # 2, 9 --> 9, 2
		# resahpe rpn_cls_score to prepare for softmax

		# softmax
		rpn_cls_prob = tf.nn.softmax( a )
		rpn_cls_prob = tf.transpose(a, [0,1,2,4,3])
		# reshape back to the original
		rpn_cls_prob = tf.reshape( rpn_cls_prob, (shape[0], shape[1], shape[2], -1) )

		return rpn_cls_prob
