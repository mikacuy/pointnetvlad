import tensorflow as tf
import numpy as np
import math
import sys
import os

#Taken from Charles Qi's pointnet code
import tf_util
from transform_nets import input_transform_net, feature_transform_net

#Adopted from Antoine Meich
import loupe as lp



def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None):
    """PointNetVLAD,    INPUT is batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X 3, 
                        OUTPUT batch_num_queries X num_pointclouds_per_query X output_dim """
    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value
    CLUSTER_SIZE=64
    OUTPUT_DIM=256
    point_cloud = tf.reshape(point_cloud, [batch_num_queries*num_pointclouds_per_query, num_points,3])

    with tf.variable_scope('transform_net1') as sc:
        input_transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, input_transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        feature_transform = feature_transform_net(net, is_training, bn_decay, K=64)
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    NetVLAD = lp.NetVLAD(feature_size=1024, max_samples=num_points, cluster_size=CLUSTER_SIZE, 
                    output_dim=OUTPUT_DIM, gating=True, add_batch_norm=True,
                    is_training=is_training)

    net= tf.reshape(net,[-1,1024])
    net = tf.nn.l2_normalize(net,1)
    output = NetVLAD.forward(net)
    print(output)

    #normalize to have norm 1
    output = tf.nn.l2_normalize(output,1)
    output =  tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM])

    return output


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        #batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
        best_pos=tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        #best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos



##########Losses for PointNetVLAD###########

#Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

#Lazy variant
def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss


def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_max(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss 

def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss  





