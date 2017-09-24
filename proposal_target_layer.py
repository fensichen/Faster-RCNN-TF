import numpy as np
import tensorflow as tf
import bbox_overlaps
import bbox_transform
import bbox_overlaps

def proposal_target_layer(rpn_rois, gt_boxes, num_classes):

	
	rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func( proposal_target_layer_py, [ rpn_rois, gt_boxes, num_classes], [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32])
	rois                   = tf.reshape( rois, [-1, 5],                      name = 'rois')
	labels                 = tf.convert_to_tensor( tf.cast(labels, tf.int32),name = 'labels')
	bbox_targets           = tf.convert_to_tensor( bbox_targets,             name = 'bbox_targets')
	bbox_inside_weights    = tf.convert_to_tensor( bbox_inside_weights,      name = 'bbox_inside_weights')
	bbox_outside_weights   = tf.convert_to_tensor( bbox_outside_weights,     name = 'bbox_outside_weights')
	return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
	
def proposal_target_layer_py(rpn_rois, gt_boxes, num_classes):
	"""
	Assign object detection proposals to ground-truth targets. Produces proposal classfication labels and bounding box regression targets 
	"""

	# Proposal ROIs (0, x1, y1, x2, y2) come from RPN blob
	all_rois                                        = rpn_rois
	print "all_rois.shape", all_rois.shape
	# Include ground-truth boxes in the set of candidates rois
	zeros                                           = np.zeros((gt_boxes.shape[0],1), dtype=gt_boxes.dtype)
	all_rois                                        = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:,:-1])))) # append gt_boxes at the last row of all_rois 
	all_rois 										= all_rois.astype(np.float32)
	print "all_rois.shape", all_rois.shape

	num_images                                      = 1 
	rois_per_image                                  = 128
	fg_rois_per_image                               = np.round( 0.25 * rois_per_image).astype(np.int32)

	# sample rois with classfication labels and bounding box regression targets
	print "fg_rois_per_image", fg_rois_per_image, "rois_per_image", rois_per_image
	labels, rois, bbox_targets, bbox_inside_weights = sample_rois( all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes )

	# 
	rois 											= rois.reshape(-1,5)
	labels 											= labels.reshape(-1,1)
	bbox_targets									= bbox_targets.reshape(-1, num_classes * 4)
	bbox_inside_weights 							= bbox_inside_weights.reshape(-1, num_classes * 4)
	bbox_outside_weights 							= np.array( bbox_inside_weights > 0) .astype( np.float32) 
	
	return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights



def sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
	"""
	generate a random sample of ROIs comprising foreground and background example
	"""

	# generate overlap matrix: ( #rois x #gt_boxes)
	overlaps 			  							= bbox_overlaps.bbox_overlaps(
													np.ascontiguousarray(all_rois[:,1:5], np.float),
													np.ascontiguousarray(gt_boxes[:,:4],  np.float)) 

	gt_assignment         							= overlaps.argmax(axis=1)   # find the gt_box index that generates max_overlaps for each roi
	max_overlaps          							= overlaps.max(axis=1)      # find the maximum overlap for each roi
	labels                							= gt_boxes[gt_assignment,4] # generate the label for each roi, according to max_overlap gt_box
	
	# select foreground ROI as thos with >= FG_Threshold overlap
	fg_inds               							= np.where(max_overlaps >= 0.5 )[0] # return the row indicies 
	# guard against the case when an image has fewer than fg_rois_per_image
	fg_rois_per_this_image 							= min(fg_rois_per_image, fg_inds.size)
	
	if fg_inds.size > 0:
		fg_inds = np.random.choice(fg_inds, fg_rois_per_this_image, replace=False)

	
	# select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI]
	bg_inds               							= np.where((max_overlaps < 0.5) & (max_overlaps >= 0.0 ))[0]
	# compute number of background ROI to take from this image
	bg_rois_per_this_image                          = rois_per_image -fg_rois_per_image
	bg_rois_per_this_image                          = min(bg_rois_per_this_image, bg_inds.size)
	
	if bg_inds.size > 0:
		bg_inds                                     = np.random.choice(bg_inds, bg_rois_per_this_image, replace=False)
	
	# the indices that we'are selecting
	keep_inds             							= np.append(fg_inds, bg_inds)
	# select sampled values from various array
	labels                				            = labels[keep_inds]
	# clamp labels for the bg ROI to 0
	labels[fg_rois_per_this_image:]					= 0
	rois                                            = all_rois[keep_inds]
	
	# bbox_target_data [ class_label, x, y, w, h]
	bbox_target_data                                = compute_targets(rois[:,1:5], gt_boxes[ gt_assignment[keep_inds], :4], labels)
	bbox_targets, bbox_inside_weights 				= get_bbox_regression_labels( bbox_target_data, num_classes) 

	print "bbox_targets.shape", bbox_targets.shape
	print "bbox_inside_weights.shape", bbox_inside_weights.shape
	return labels, rois, bbox_targets, bbox_inside_weights


def compute_targets(ex_rois, gt_rois, labels):
	"""
	compute bounding-box regression targets for an image
	"""

	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 4

	targets                 = bbox_transform.bbox_transform(ex_rois, gt_rois)
	return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def get_bbox_regression_labels(bbox_target_data, num_classes):

	"""
	bbox_target_data: N x ( class_label, tx, ty, tw, th) which is N x 5 array, K = #class_label
	Returns: 
		bbox_target         : N x 4K blob of regression targets, K is num_classes
		bbox_inside_weights : N x 4K blob of loss weights 
	"""
	clss 					= bbox_target_data[:,0] # class label
	bbox_targets            = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)	
	bbox_inside_weights     = np.zeros(bbox_targets.shape, dtype=np.float32)	
	inds                    = np.where(clss > 0)[0] # find the foreground index

	for ind in inds:
		cls 				                = clss[ind]
		start 				                = int(4*cls)
		end                                 = start+4 
		bbox_targets[ind, start:end]        = bbox_target_data[ind, 1:]
		bbox_inside_weights[ind, start:end] = (1,1,1,1)
		
	return bbox_targets, bbox_inside_weights 


