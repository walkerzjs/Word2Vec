import tensorflow as tf
import numpy as np
import tarfile
import re

batch_size = 80
max_col_size=40


def cleanSplitSentences(string):
    string = string.lower().replace("<br />", " ")
    string = re.sub("[^A-Za-z0-9 \!\?\.]+", " ", string)
    string = re.sub("([\!\?\.])[\!\?\.]+", r"\1", string)
    string = re.sub("\s+", " ", string)
    string = re.split("(\W)", string)
    string = [s for s in string if not (re.match("\s", s) or s=="")]
    return string



def makeIndexList(content, glove_dict, max_col_size):
    out = list()
    for i in range(max_col_size):
        if i < len(content):
            word = content[i]
            if word in glove_dict.keys():
                out.append(glove_dict[word]) #index of word
            else:
                out.append(glove_dict['UNK']) #unknown word index
        else:
            out.append(glove_dict['_']) #empty space padding index
    
    return out


def load_data(glove_dict, max_col_size = max_col_size):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    filename = 'reviews.tar.gz'
    
    data = list()
    with tarfile.open(filename, "r") as tar:
        tar_list = tar.getnames()
        pos_files = [tarinfo for tarinfo in tar_list if tarinfo.startswith("pos/")]
        neg_files = [tarinfo for tarinfo in tar_list if tarinfo.startswith("neg/")]
        
        
        # retrive and clean data from text files, split into a list of words
        for file_name in pos_files + neg_files:
            file = tar.extractfile(file_name)
            if file:
                content = file.read().decode()
                content = cleanSplitSentences(content)
                index_list = makeIndexList(content, glove_dict, max_col_size)
                data.append(index_list)
                
    return np.array(data, dtype=int)


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    
    outWords = dict()
#    invWords = dict()
    outArray = list()
    with open("glove.6B.50d.txt",'r',encoding="utf-8") as f:
        for i, line in enumerate (f.readlines()):
            line = line.strip().split(" ")
            word = line.pop(0)
            outWords[word] = i
#            invWords[i] = word 
            outArray.append(line)
        
        #unknown word placeholder, array of 40 zeros
        i+=1
        word = 'UNK'
        outWords[word] = i
#        invWords[i] = word 
        outArray.append(['0.0',]*len(outArray[0]))
        
    return np.array(outArray, dtype=np.float32), outWords



def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    input_data = tf.placeholder(tf.int32, shape=[batch_size, max_col_size])
    labels = tf.placeholder(tf.float32, shape=[batch_size, 2])
    embed = tf.Variable(tf.zeros([batch_size, max_col_size, glove_embeddings_arr.shape[1]]),dtype=tf.float32)
    embed = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    
    
    with tf.variable_scope('lstm15'):
#        dropout_keep_prob = tf.placeholder(tf.float32,shape=(),name='dropout_keep_prob')
        dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
        
        # define lstm model for each layer
        def lstm():
            
            lstm = tf.contrib.rnn.BasicLSTMCell(60)
            lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob, input_keep_prob=1, state_keep_prob=1)
            return lstm
        
        # define multilayer model
        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(2)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        
        output, state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32,initial_state = initial_state)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        
        dim =cell.output_size
        out_dim = labels.get_shape()[1].value
        
        #create weights of new neurons in the first layer
        W = tf.Variable(tf.truncated_normal([dim,out_dim], stddev=0.1,dtype=tf.float32))
        B = tf.Variable(tf.constant(0.1,shape=[out_dim],dtype=tf.float32))
        
        #compute results
        logits = tf.matmul(last,W)+B
#        preds = tf.nn.softmax(logits)
        batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        
        #define loss function an optimizer
        loss = tf.reduce_mean(batch_xentropy)
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
        prediction_checked = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(prediction_checked, tf.float32))

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss



