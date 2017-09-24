#anchor_target_layer.py
import numpy as np
import random
import tensorflow as tf
import generate_anchors

import pyximport 
pyximport.install()

import bbox_overlaps
import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_dims, feat_strides, anchor_scales):

	rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
		tf.py_func( anchor_target_layer_python,[rpn_cls_score, gt_boxes, im_dims, feat_strides, anchor_scales],
		[tf.float32, tf.float32, tf.float32, tf.float32] )
	

	rpn_labels                 = tf.convert_to_tensor( tf.cast( rpn_labels, tf.int32), name = "rpn_labels"              )
	rpn_bbox_targets           = tf.convert_to_tensor( rpn_bbox_targets              , name = "rpn_bbox_targets"        )
	rpn_bbox_inside_weights    = tf.convert_to_tensor( rpn_bbox_inside_weights       , name = "rpn_bbox_inside_weights" )
	rpn_bbox_outside_weights   = tf.convert_to_tensor( rpn_bbox_outside_weights      , name = "rpn_bbox_outside_weights")

	return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def anchor_target_layer_python(rpn_cls_score, gt_boxes, im_dims, feat_strides, anchor_scales):
	"""
	Produce anchor classfication labels and bounding-box regression targets
	
	# for each (H,W ) in location i 
	#	generate 9 anchor boxes centered on cell i
	#	apply predicted bbox deltas at cell i to each of the 9 anchors
	# filter out-of-image anchors
	# measure ground-truth overlapping

	"""
	allowed_border             = 0            
	im_dims                    = im_dims[0] 
	anchor_scales 			   = np.array( anchor_scales)
	anchors       			   = generate_anchors.generate_anchors( base_size = 16, ratios=[0.5, 1, 2], scales = anchor_scales )
	num_anchors                = anchors.shape[0]
	
	# find the shape ( ..., H, W)
	height        			   = rpn_cls_score.shape[1]
	width         			   = rpn_cls_score.shape[2]
	

	DEBUG = 1
	if DEBUG:  
		print ''  
		print 'im_size: ({}, {})'.format(im_dims[0], im_dims[1])  
		print 'scale: {}'.format(anchor_scales)  
		print 'height, width: ({}, {})'.format(height, width)  
		print 'rpn: gt_boxes.shape ({})'.format(gt_boxes.shape)  

	# step1: generate proposal from bbox deltas and shifted anchors 
	# shift_x and shift_y are each pixel index in [ height, width ] 	
	shift_x 	  			   = np.arange(0, width ) * feat_strides 
	shift_y                    = np.arange(0, height) * feat_strides
	shift_x, shift_y           = np.meshgrid( shift_x, shift_y )
	shifts                     = np.vstack( ( shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel() ) )
	shifts 					   = shifts.transpose()
	
	#move anchor according to shifts  
	K 						   = shifts.shape[0]	                        # number of candidate location 
	b 						   = anchors.reshape((1, num_anchors, 4 ))      # ( 1, A, 4)
	c 						   = shifts.reshape((1, K, 4)).transpose(1,0,2) # ( K, 1, 4)
	all_anchors 			   = b + c
	all_anchors                = all_anchors.reshape( ( K * num_anchors, 4 ) ) # ( K*A, 4)
	total_anchors              = int( K * num_anchors )

	
	# find anchors inside the image
	if im_dims is not None:
		inds_inside            = np.where( 
							   ( all_anchors[:,0] >= 0 ) & 
							   ( all_anchors[:,1] >= 0 ) & 
							   ( all_anchors[:,2] <  im_dims[1] +0 ) &   # width
							   ( all_anchors[:,3] <  im_dims[0])+0)[0]   # take the row index

	if DEBUG:
		print "total_anchors", total_anchors
		
	# keep only inside anchors
	anchors                    = all_anchors[inds_inside,:]
	if DEBUG:
		print "anchors.shape", anchors.shape

	# step2: assign label for each anchor in anchors
	# label: 1 is postive, 0 is native, -1 is don't care
	labels                     = np.empty( ( len(inds_inside),  ), dtype = np.float32)
	labels.fill(-1)
	
	# overlap between the anchors and the gt boxes
	overlaps                   = bbox_overlaps.bbox_overlaps(
							     np.ascontiguousarray(anchors, dtype = np.float), 
							     np.ascontiguousarray(gt_boxes, dtype = np.float)		
							)  # ( #inds_inside x gt_boxes.shape[0])


	argmax_overlaps            = overlaps.argmax(axis = 1 ) # for each anchor( for each row), find its max overlap gt box index among all the gt_boxes 
	max_overlaps               = overlaps[ np.arange( len(inds_inside) ), argmax_overlaps] # [ inds_inside, 1], return the max_overlap area for each anchor	
	gt_argmax_overlaps         = overlaps.argmax(axis = 0) # for each gt_box, find its max overlap against all the anchors, it return the anchor index
	gt_max_overlaps            = overlaps[ gt_argmax_overlaps, np.arange(overlaps.shape[1])] # for each ground truth, return it maximum overlap     
	gt_argmax_overlaps         = np.where(overlaps == gt_max_overlaps)[0]					 # return all the anchors index which has maximun overlap
	
	# set fg and bg label for each anchor in anchors
	labels[max_overlaps < 0.3] = 0 # set labels of those anchor which max_overlaps < 0.3
	labels[gt_argmax_overlaps] = 1 # for each gt, set anchor with highest overlap to 1  
	labels[max_overlaps > 0.7] = 1 # set labels of those anchor which max_overlaps < 0.3

	# subsample positive labels if there are too many
	num_fg                     = int( 0.5 * 256 )
	fg_inds                    = np.where(labels == 1)[0]
	if len(fg_inds ) > num_fg:
		disable_inds           = np.random.choice( fg_inds, size=(len(fg_inds) - num_fg), replace = False )
		labels[disable_inds]   = -1 # set it to don't care

	# subsample negative lables if there are too many
	num_bg                     = 256 - np.sum(labels == 1)
	bg_inds                    = np.where(labels == 0)[0]
	if len(bg_inds) > num_bg:
		disable_inds 		   = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace = False)
		labels[disable_inds]   = -1


	# bbox targets : the deltas (relative to anchors) that faster R-CNN should try to predict at each anchor
	bbox_targets               = np.zeros( (len(inds_inside), 4), dtype = np.float32 )
	bbox_targets               = compute_target( anchors, gt_boxes[argmax_overlaps,:])

	# it is used for p*_{i} in cost function (equation 1): Inside weights is for specifying anchors or rois with positive labe
	bbox_inside_weights        = np.zeros( (len(inds_inside), 4), dtype = np.float32 )
	bbox_inside_weights[labels == 1] = np.array((1.0, 1.0, 1.0, 1.0)) 
	bbox_outside_weights       = np.zeros( (len(inds_inside), 4), dtype = np.float32)
	
	# uniform weight per sample : Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p
	num_examples               = np.sum(labels >= 0)
	positive_weights           = np.ones((1,4)) * 1.0/num_examples 
	negative_weights           = np.ones((1,4)) * 1.0/num_examples

	bbox_outside_weights[labels == 1,: ] = positive_weights
	bbox_outside_weights[labels == 0,: ] = negative_weights

	# map to oriignal set of anchors
	labels                     = unmap(labels,              total_anchors, inds_inside, fill = -1)
	bbox_targets               = unmap(bbox_targets       , total_anchors, inds_inside, fill = 0 )
	bbox_inside_weights        = unmap(bbox_inside_weights, total_anchors, inds_inside, fill = 0 )
	bbox_outside_weights       = unmap(bbox_outside_weights,total_anchors, inds_inside, fill = 0 )

	# labels
	labels                     = labels.reshape((1, height, width, num_anchors)).transpose(0, 3, 1, 2)
	labels                     = labels.reshape((1,1, num_anchors * height* width))
 	rpn_labels 			       = labels
	rpn_bbox_targets 		   = bbox_targets.reshape        ((1, height, width, num_anchors * 4)).transpose(0,3,1,2)
	rpn_bbox_inside_weights    = bbox_inside_weights.reshape ((1, height, width, num_anchors * 4)).transpose(0,3,1,2)
	rpn_bbox_outside_weights   = bbox_outside_weights.reshape((1, height, width, num_anchors * 4)).transpose(0,3,1,2)

	if DEBUG:
		print "rpn_labels.shape.shape", rpn_labels.shape
		print "rpn_bbox_targets.shape", rpn_bbox_targets.shape
		print "rpn_bbox_inside_weights.shape", rpn_bbox_inside_weights.shape
		print "rpn_bbox_outside_weights.shape",rpn_bbox_outside_weights.shape

	return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights  

def unmap( data, count, inds, fill = 0):
	"""
	Unmap a subset of item (data) back to the original set of items (of size count)
	"""
	if len(data.shape) == 1 :
		ret = np.empty( (count,), dtype= np.float32)
		ret.fill(fill)
		ret[inds] = data
	else:
		ret         = np.empty( (count, ) + data.shape[1:], dtype = np.float32)
		ret.fill(fill)
		ret[inds,:] = data

	return ret

def compute_target(ex_rois, gt_rois):
	"""Compute bounding-box regression targets for an image."""
	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 5

	return bbox_transform.bbox_transform(ex_rois, gt_rois[:,:4].astype(np.float32, copy = False) )