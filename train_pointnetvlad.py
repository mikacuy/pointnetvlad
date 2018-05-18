import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pointnetvlad_cls import *
from loading_pointclouds import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree


#params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--positives_per_query', type=int, default=2, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18, help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Initial learning rate [default: 0.00005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2, help='Margin for hinge loss [default: 0.2]')
FLAGS = parser.parse_args()

BATCH_NUM_QUERIES = FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY= FLAGS.positives_per_query
NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MARGIN1 = FLAGS.margin_1
MARGIN2 = FLAGS.margin_2

TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
TEST_FILE = 'generating_queries/test_queries_baseline.pickle'

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#Load dictionary of training queries
TRAINING_QUERIES= get_queries_dict(TRAIN_FILE)
TEST_QUERIES= get_queries_dict(TEST_FILE)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

global HARD_NEGATIVES
HARD_NEGATIVES={}

global TRAINING_LATENT_VECTORS
TRAINING_LATENT_VECTORS=[]

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

#learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def train():
    global HARD_NEGATIVES
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("In Graph")
            query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            other_negatives=  placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            batch = tf.Variable(0)
            epoch_num = tf.placeholder(tf.float32, shape=())
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, positives, negatives, other_negatives],1)
                print(vecs)                
                out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
                print(out_vecs)
                q_vec, pos_vecs, neg_vecs, other_neg_vec= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY,1],1)
                print(q_vec)
                print(pos_vecs)
                print(neg_vecs)
                print(other_neg_vec)

            #loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
            #loss = softmargin_loss(q_vec, pos_vecs, neg_vecs)
            #loss = quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(epoch_num)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Initialize a new model
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized")

        # Restore a model
        # saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt"))
        # print("Model restored.")


        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'other_negatives': other_negatives,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'epoch_num': epoch_num,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs,
               'other_neg_vec': other_neg_vec}


        for epoch in range(MAX_EPOCH):
            print(epoch)
            print()
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver)          



def train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS

    is_training = True
    sampled_neg=4000
    #number of hard negatives in the training tuple
    #which are taken from the sampled negatives
    num_to_take=10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
        batch_keys= train_file_idxs[i*BATCH_NUM_QUERIES:(i+1)*BATCH_NUM_QUERIES]
        q_tuples=[]

        faulty_tuple=False
        no_other_neg=False
        for j in range(BATCH_NUM_QUERIES):
            if(len(TRAINING_QUERIES[batch_keys[j]]["positives"])<POSITIVES_PER_QUERY):
                faulty_tuple=True
                break

            #no cached feature vectors               
            if(len(TRAINING_LATENT_VECTORS)==0):
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif(len(HARD_NEGATIVES.keys())==0):
                query=get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives=TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs= get_random_hard_negatives(query, negatives, num_to_take)
                print(hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query=get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives=TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs= get_random_hard_negatives(query, negatives, num_to_take)
                hard_negs= list(set().union(HARD_NEGATIVES[batch_keys[j]], hard_negs))
                print('hard',hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))           
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))           
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            
            if(q_tuples[j][3].shape[0]!=NUM_POINTS):
                no_other_neg= True
                break

        #construct query array
        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue            

        queries=[]
        positives=[]
        negatives=[]
        other_neg=[]
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries= np.array(queries)
        queries= np.expand_dims(queries,axis=1)
        other_neg= np.array(other_neg)
        other_neg= np.expand_dims(other_neg,axis=1)
        positives= np.array(positives)
        negatives= np.array(negatives)
        log_string('----' + str(i) + '-----')
        if(len(queries.shape)!=4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue
        
        feed_dict={ops['query']:queries, ops['positives']:positives, ops['negatives']:negatives, ops['other_negatives']:other_neg, ops['is_training_pl']:is_training, ops['epoch_num']:epoch}
        summary, step, train, loss_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        log_string('batch loss: %f' % loss_val)

        if(i%200==7):
            test_file_idxs = np.arange(0,len(TEST_QUERIES.keys()))
            np.random.shuffle(test_file_idxs)

            eval_loss=0
            eval_batches=5
            eval_batches_counted=0
            for eval_batch in range(eval_batches):
                eval_keys= test_file_idxs[eval_batch*BATCH_NUM_QUERIES:(eval_batch+1)*BATCH_NUM_QUERIES]
                eval_tuples=[]

                faulty_eval_tuple=False
                no_other_neg= False
                for e_tup in range(BATCH_NUM_QUERIES):
                    if(len(TEST_QUERIES[eval_keys[e_tup]]["positives"])<POSITIVES_PER_QUERY):
                        faulty_eval_tuple=True
                        break
                    eval_tuples.append(get_query_tuple(TEST_QUERIES[eval_keys[e_tup]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TEST_QUERIES, hard_neg=[], other_neg=True)) 

                    if(eval_tuples[e_tup][3].shape[0]!=NUM_POINTS):
                        no_other_neg= True
                        break

                if(faulty_eval_tuple):
                    log_string('----' + 'FAULTY EVAL TUPLE' + '-----')
                    continue

                if(no_other_neg):
                    log_string('----' + str(i) + '-----')
                    log_string('----' + 'NO OTHER NEG EVAL' + '-----')
                    continue  

                eval_batches_counted+=1
                eval_queries=[]
                eval_positives=[]
                eval_negatives=[]
                eval_other_neg=[]

                for tup in range(len(eval_tuples)):
                    eval_queries.append(eval_tuples[tup][0])
                    eval_positives.append(eval_tuples[tup][1])
                    eval_negatives.append(eval_tuples[tup][2])
                    eval_other_neg.append(eval_tuples[tup][3])

                eval_queries= np.array(eval_queries)
                eval_queries= np.expand_dims(eval_queries,axis=1)                
                eval_other_neg= np.array(eval_other_neg)
                eval_other_neg= np.expand_dims(eval_other_neg,axis=1)
                eval_positives= np.array(eval_positives)
                eval_negatives= np.array(eval_negatives)
                feed_dict={ops['query']:eval_queries, ops['positives']:eval_positives, ops['negatives']:eval_negatives, ops['other_negatives']:eval_other_neg, ops['is_training_pl']:False, ops['epoch_num']:epoch}
                e_summary, e_step, e_loss= sess.run([ops['merged'], ops['step'], ops['loss']], feed_dict=feed_dict)
                eval_loss+=e_loss
                if(eval_batch==4):
                    test_writer.add_summary(e_summary, e_step)
            average_eval_loss= float(eval_loss)/eval_batches_counted
            log_string('\t\t\tEVAL')
            log_string('\t\t\teval_loss: %f' %average_eval_loss)


        if(epoch>5 and i%700 ==29):
            #update cached feature vectors
            TRAINING_LATENT_VECTORS=get_latent_vectors(sess, ops, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        if(i%3000==101):
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


def get_feature_representation(filename, sess, ops):
    is_training=False
    queries=load_pc_files([filename])
    queries= np.expand_dims(queries,axis=1)
    if(BATCH_NUM_QUERIES-1>0):
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
    else:
        q=queries
    fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
    fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
    fake_other_neg=np.zeros((BATCH_NUM_QUERIES,1,NUM_POINTS,3))
    feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['other_negatives']: fake_other_neg, ops['is_training_pl']:is_training}
    output=sess.run(ops['q_vec'], feed_dict=feed_dict)
    output=output[0]
    output=np.squeeze(output)
    return output

def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs=[]
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])
    
    latent_vecs=np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]),k=num_to_take)
    hard_negs=np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs= hard_negs.tolist()
    return hard_negs

def get_latent_vectors(sess, ops, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY+1)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        file_names=[]
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries=load_pc_files(file_names)

        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))

        q4=queries[BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+2)]
        q4=np.expand_dims(q4,axis=1)

        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3,ops['other_negatives']:q4, ops['is_training_pl']:is_training}
        o1, o2, o3, o4=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec']], feed_dict=feed_dict)
        
        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))
        o4=np.reshape(o4,(-1,o4.shape[-1]))        

        out=np.vstack((o1,o2,o3,o4))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):  
        q_output=q_output.reshape(-1,q_output.shape[-1])

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        index=train_file_idxs[q_index]
        queries=load_pc_files([dict_to_process[index]["query"]])
        queries= np.expand_dims(queries,axis=1)

        if(BATCH_NUM_QUERIES-1>0):
            fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
            q=np.vstack((queries,fake_queries))
        else:
            q=queries

        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        fake_other_neg=np.zeros((BATCH_NUM_QUERIES,1,NUM_POINTS,3))
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['other_negatives']:fake_other_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    print(q_output.shape)
    return q_output

if __name__ == "__main__":
    train()
