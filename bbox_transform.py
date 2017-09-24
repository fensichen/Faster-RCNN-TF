import numpy as np

def bbox_transform(ex_rois, gt_rois):
	"""
	Receive two sets of bounding boxes, denoted by two opposite corners (x1, y1, x2, y2) and returns the target deltas [tx, ty, tw, th] 
	# that Faster R-CNN should aim for.  
	# ex_rois: anchor_boxs 
	# gt_rois: the corresponding ground truth bounding box 
	# This function implement equation (2) in Faster-RCNN paper 
	"""
	ex_widths          =  ex_rois[:,2] - ex_rois[:,0] + 1.0
	ex_heights         =  ex_rois[:,3] - ex_rois[:,1] + 1.0
	ex_ctr_x           =  ex_rois[:,0] + 0.5 * ex_widths
	ex_ctr_y           =  ex_rois[:,1] + 0.5 * ex_heights

	gt_widths          =  gt_rois[:,2] + gt_rois[:,0] + 1.0
	gt_heights         =  gt_rois[:,3] + gt_rois[:,1] + 1.0 
	gt_ctr_x           =  gt_rois[:,0] + 0.5 * gt_widths
	gt_ctr_y           =  gt_rois[:,1] + 0.5 * gt_heights

	target_dx          =  (gt_ctr_x - ex_ctr_x ) / ex_widths 
	target_dy          =  (gt_ctr_y - ex_ctr_y ) / ex_heights
	target_dw          =  np.log(gt_widths/ ex_widths)
	target_dh          =  np.log(gt_heights/ex_heights)

	targets            =  np.vstack( (target_dx, target_dy, target_dw, target_dh ) )
	targets            =  targets.transpose()
	return targets

def bbox_transform_inv(boxes, deltas):
	"""
	Applied deltas to box cooridate to obtain new boxes
	input: bboxes is in the form of [x1,y1, x2, y2], where (Ax,Ay) is the top-left corner coordinate, and Aw and Ah are bounding box width and height
		
	output: 
	 predicted boxes
	"""
	if boxes.shape[0] == 0:
		return np.zeros( (0, deltas[1]), dtype = deltas.dtype )

	boxes 			   = boxes.astype(deltas.dtype, copy = False)
	# width, height, center of bounding box 
	widths             = boxes[:,2] - boxes[:,0] + 1.0
	heights            = boxes[:,3] - boxes[:,1] + 1.0
	ctr_x              = boxes[:,0] + 0.5 * widths
	ctr_y              = boxes[:,1] + 0.5 * heights

	# delta
	dx                 = deltas[:,0::4]
	dy                 = deltas[:,1::4]
	dw                 = deltas[:,2::4]
	dh                 = deltas[:,3::4]

	# pred = bounding box + delta 
	pred_ctr_x         = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
	pred_ctr_y         = dy * heights[:,np.newaxis] + ctr_y[:, np.newaxis]
	pred_w             = np.exp(dw) * widths[:,np.newaxis]
	pred_h             = np.exp(dh) * heights[:,np.newaxis]


	pred_boxes         = np.zeros(deltas.shape, dtype = deltas.dtype)
	# x1
	pred_boxes[:,0::4] = pred_ctr_x - 0.5 * pred_w 
	# y1 
	pred_boxes[:,1::4] = pred_ctr_y - 0.5 * pred_h
	# x2 
	pred_boxes[:,2::4] = pred_ctr_x + 0.5 * pred_w 
	# y2 
	pred_boxes[:,3::4] = pred_ctr_y + 0.5 * pred_h

	return pred_boxes


def clip_boxes( boxes, im_shape):
	"""
	clip boxes to image boundaries
	input:
	im_shape[0] : height
	im_shape[1] : width

	"""
	# x1 >= 0
	
	boxes[:,0::4]      = np.maximum( np.minimum(boxes[:, 0::4], im_shape[0,1]-1 ), 0)
	# y1 >= 0
	boxes[:,1::4]      = np.maximum( np.minimum(boxes[:, 1::4], im_shape[0,0]-1 ), 0)
	# x2 < im_shape[1]
	boxes[:,2::4]      = np.maximum( np.minimum(boxes[:, 2::4], im_shape[0,1]-1 ), 0)
	# y2 < im_shape[0]
	boxes[:,3::4]      = np.maximum( np.minimum(boxes[:, 3::4], im_shape[0,0]-1 ), 0)
	return boxes
