import tensorflow as tf
import numpy as np
import os, sys
import random
import scipy as scp
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

label_dimension = 5
class_table     = ['Background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

class DataHandler(object):
    def __init__(self):
        self.image_list     = []
        self.label_list     = []
        self.VGG_MEAN       = [ 123.68, 116.779, 103.939] # RGB

    def num( self ):
        return len(self.image_list)

    def get_file_list(self, path_image, path_label = None):
        image_list          = [line.strip().split(' ')[0] for line in open( path_image )]
        self.image_list     = self.image_list + image_list 

        if path_label is not None:
            label_list          = [line.strip().split(' ')[0] for line in open( path_label )]
            self.label_list     = self.label_list + label_list
        
        self.perm           = range( len(self.image_list) )

    def shuffle(self):    
        random.shuffle( self.perm ) # in-place shuffle 

    def apply_hflip_to_gtbox(self, gt_box, im_dims):
        x1                   = gt_box[0]
        y1                   = gt_box[1]
        x2                   = gt_box[2]
        y2                   = gt_box[3] 
        score                = gt_box[4] 
        x1_tmp               = im_dims[1] - 1 - x2
        x2_tmp               = im_dims[1] - 1 - x1
        return np.stack((x1_tmp,y1,x2_tmp,y2,score))

    def load_data(self, start, end, is_training = True ):
        """
        Args: 
            start index
            end index
        Return:
            image array
            label array
        """
        perm                 = self.perm
        probe                = scp.misc.imread( self.image_list[ perm[start]] )

        HEIGHT               = probe.shape[0]
        WIDTH                = probe.shape[1]
        CHANNEL              = probe.shape[2]
        
        batch_size           = end - start
        data                 = np.zeros( (batch_size, HEIGHT, WIDTH, CHANNEL), dtype = np.float32 )
        #labels               = np.zeros( (0, label_dimension       ), dtype = np.float32 )
        labels               = []
        im_dims              = np.zeros( (batch_size, 2)                     , dtype = np.int64 )
        
        for index in range( start, end ):
            
            img              = scp.misc.imread( self.image_list[ self.perm[ start + index ] ] )
            img              = img.astype( np.float32 )
            #print "loading ", self.image_list[ self.perm[ start + index ] ]
            if ( img.shape == probe.shape ) == False:
                img = np.resize( img, probe.shape )

            if is_training == True:
                print "self.label_list[ ",  self.perm[ start + index] , " ] ", self.label_list[ self.perm[ start + index ] ]
                label_all        = [line.strip().split(' ') for line in open( self.label_list[ self.perm[ start + index ] ] )]  
                num_of_object    = len(label_all)

                for i in range(0, num_of_object):
                    class_id     = class_table.index(label_all[i][0])
                    if class_id == 8 or class_id == 9:
                        continue
                    
                    coordinate   = label_all[i][4:8]
                    coordinate.append(class_id)
                    result       = [float(c) for c in coordinate]
                    result       = np.array(result, dtype = np.int32)
                    labels.append(result)
            
            # Subtract average of each color channel to center the data around zero mean for each channel (R,G,B). 
            # This typically helps the network to learn faster since gradients act uniformly for each channel
        
            if is_training == False:
                hflip = 0
            else: 
                hflip = random.randint(0,1)
               
                if hflip == 1:
                    img    = img[:,::-1,:]
                    
                    labels_flip = []
                    for i in range(0, len(labels)):
                        tmp  = self.apply_hflip_to_gtbox(labels[i], img.shape[:2])                    
                        labels_flip.append(tmp)
                        
                    labels = labels_flip
                   
            img                  = img - self.VGG_MEAN
            data[  index -start] = img
            labels               = np.array(labels)           
            im_dims[index-start] = img.shape[:2]


        # create tensorflow feed dictionary   
        if is_training == False:
            return data, im_dims
        else:
            return data, labels, im_dims


if __name__ == '__main__':
    data_handler = DataHandler()
    data_handler.get_file_list( '/home/fensi/nas/KITTI_OBJECT/train.txt', '/home/fensi/nas/KITTI_OBJECT/label.txt' )
    data, labels = data_handler.load_data(1,10)
    print data.shape, labels.shape