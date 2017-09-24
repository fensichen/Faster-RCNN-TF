import tensorflow as tf

def roi_pool(feature_map, rois, im_dims):
	"""
	For each object proposal, a region of interest pooling layer extracts a fixed lengh feature vector from the feature map 

	ROI from RPN are formatted as : ( image_id, x1, y1, x2, y2 )
	scene mini-batches are sampled from a single image, image_id = 0

	ROI pooling layer uses max pooling to convert the features inside valid ROI into a small feature map with a fixed spatial extent of H x W
	"""

	with tf.variable_scope('roi_pool'):
		# image that the ROI is taken from 
		box_ind         = tf.cast(rois[:,0], dtype=tf.int32) # 1-D tensor of shape [num_boxes]. The value of box_ind[i] specified the image that the i-th box refers to. 
		print "box_ind", box_ind
		# roi box coordinates (x1, y1, x2, y2). Must be normalized and orderd to [y1, x1, y2, x2 ]
		boxes           = rois[:,1:]
		normalization   = tf.cast( tf.stack( [im_dims[:,1], im_dims[:,0], im_dims[:,1], im_dims[:, 0]] , axis =1 ), dtype = tf.float32)
		boxes           = tf.div(boxes, normalization) 
		boxes           = tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis = 1) # y1, x1, y2, x2 -> to fit tf.image.crop_and_resize

		# roi pool output size
		crop_size       = tf.constant([14,14])
		# ROI pool 
		pooledfeatures  = tf.image.crop_and_resize(image=feature_map, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
		# Max pool to (7x7)
		pooledfeatures  = tf.nn.max_pool(pooledfeatures, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	return pooledfeatures 