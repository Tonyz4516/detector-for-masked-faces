from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import time


def main(data, model, image_size=160, seed=666):
    print("all start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with tf.Graph().as_default():
      
        with tf.compat.v1.Session() as sess:
            
            np.random.seed(seed=seed)
            
            # Load the model
#             print('Loading feature extraction model')
#             print("LOAD FACENET start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            facenet.load_model(model)
#             print("LOAD FACENET END, tensor start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
#             print("tensor end, embedding start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # Run forward pass to calculate embeddings
#             print('Calculating features for images')
            emb_array = np.zeros((1, embedding_size))
#             print("embed1 end, embedding start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            images = facenet.load_data(data, False, False, image_size)
#             print("embed2 end, embedding start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
#             print("embed3 end, embedding start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
#             print("embedding end:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            return emb_array
