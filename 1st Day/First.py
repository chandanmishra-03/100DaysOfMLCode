import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler

data = pd.read_csv('data/heart.csv')
data.head()


data.info()
data.shape

dummies = pd.get_dummies(data['famhist'],prefix='famhist', drop_first=False)
data = pd.concat([data,dummies], axis=1)
data.head()

data = data.drop(['famhist'], axis=1)
data.head()

# from sklearn.preprocessing import StandardScaler

inputs=['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']

labels = data['chd']
# min-max scaling
for each in inputs:
    data[each] = ( data[each] - data[each].min() ) / data[each].max()
    
print(data.head())
print(labels.shape)

features = data.drop(['chd'], axis=1)
features.head()

features, labels = np.array(features), np.array(labels)
print(len(features), len(labels))

##Spliting into training and testing data

# fraction of examples to keep for training
split_frac = 0.8
n_records = len(features)
split_idx = int(split_frac*n_records)

train_X, train_Y = features[:split_idx], labels[:split_idx]
test_X, test_Y = features[split_idx:], labels[split_idx:]

##Building Tensorflow Model
n_labels= 2
n_features = 10

#hyperparameters

learning_rate = 0.1
n_epochs= 200
n_hidden1 = 5
# batch_size = 128
# display_step = 1
def build_model():
    
    tf.reset_default_graph()

    inputs = tf.placeholder(tf.float32,[None, 10], name ='inputs' )
    labels = tf.placeholder(tf.int32, [None,], name='output')
    labels_one_hot = tf.one_hot(labels, 2)
    
    weights = {
        'hidden_layer': tf.Variable(tf.truncated_normal([n_features,n_hidden1], stddev=0.1)),
        'output':tf.Variable(tf.truncated_normal([n_hidden1, n_labels], stddev=0.1))
    }
    
    bias = {
        'hidden_layer':tf.Variable(tf.zeros([n_hidden1])),
        'output':tf.Variable(tf.zeros(n_labels))
    }
    
    hidden_layer = tf.nn.bias_add(tf.matmul(inputs,weights['hidden_layer']), bias['hidden_layer'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    logits = tf.nn.bias_add(tf.matmul(hidden_layer, weights['output']), bias['output'])
    
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot)
    cost = tf.reduce_mean(entropy)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        #tensorboard
        file_writer = tf.summary.FileWriter('./logs/1', sess.graph)
        
        for epoch in range(n_epochs):
            
            _, loss = sess.run([optimizer, cost], feed_dict={inputs:train_X, labels:train_Y})
           
            print("Epoch: {0} ; training loss: {1}".format(epoch, loss))
            
        print('training finished')
        
         # testing the model on test data
            
#         test_loss,logits = sess.run([loss,logits],feed_dict={inputs:test_X,labels:test_Y})
        
#         predictions = tf.nn.softmax(logits)
        
#         correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf.one_hot(test_Y), 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) 
        
#         print('model accuracy : {}'.format(accuracy))
         # Test model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({inputs: test_X, labels: test_Y}))
        
        
build_model()

##Name Scoping
def build_model_2():
    
    tf.reset_default_graph()
    
    with tf.name_scope('inputs'):

        inputs = tf.placeholder(tf.float32,[None, 10], name ='inputs' )
        
    with tf.name_scope('target_labels'):
        labels = tf.placeholder(tf.int32, [None,], name='output')
        labels_one_hot = tf.one_hot(labels, 2)
    
    with tf.name_scope('weights'):
        weights = {
            'hidden_layer': tf.Variable(tf.truncated_normal([n_features,n_hidden1], stddev=0.1), name='hidden_weights'),
            'output':tf.Variable(tf.truncated_normal([n_hidden1, n_labels], stddev=0.1), name='output_weights')
        }
    
    with tf.name_scope('biases'):
    
        bias = {
            'hidden_layer':tf.Variable(tf.zeros([n_hidden1]), name='hidden_biases'),
            'output':tf.Variable(tf.zeros(n_labels), name='output_biases')
        }
    with tf.name_scope('hidden_layers'):

        hidden_layer = tf.nn.bias_add(tf.matmul(inputs,weights['hidden_layer']), bias['hidden_layer'])
        hidden_layer = tf.nn.relu(hidden_layer, name='hidden_layer_output')
        
    with tf.name_scope('predictions'):

        logits = tf.nn.bias_add(tf.matmul(hidden_layer, weights['output']), bias['output'], name='predictions')
    
    with tf.name_scope('cost'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot, name='cross_entropy')
        cost = tf.reduce_mean(entropy, name='cost')
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        #tensorboard
        file_writer = tf.summary.FileWriter('./logs/2', sess.graph)
        
        for epoch in range(n_epochs):
            
            _, loss = sess.run([optimizer, cost], feed_dict={inputs:train_X, labels:train_Y})
           
            print("Epoch: {0} ; training loss: {1}".format(epoch, loss))
            
        print('training finished')
        
         # testing the model on test data
            
#         test_loss,logits = sess.run([loss,logits],feed_dict={inputs:test_X,labels:test_Y})
        
#         predictions = tf.nn.softmax(logits)
        
#         correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf.one_hot(test_Y), 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) 
        
#         print('model accuracy : {}'.format(accuracy))
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({inputs: test_X, labels: test_Y}))
        
model2 = build_model_2()

##Visualising weights and distributions
def build_model_3():
    
    tf.reset_default_graph()
    
    with tf.name_scope('inputs'):

        inputs = tf.placeholder(tf.float32,[None, 10], name ='inputs' )
        
    with tf.name_scope('target_labels'):
        labels = tf.placeholder(tf.int32, [None,], name='output')
        labels_one_hot = tf.one_hot(labels, 2)
    
    with tf.name_scope('weights'):
        weights = {
            'hidden_layer': tf.Variable(tf.truncated_normal([n_features,n_hidden1], stddev=0.1), name='hidden_weights'),
            'output':tf.Variable(tf.truncated_normal([n_hidden1, n_labels], stddev=0.1), name='output_weights')
        }
        
        tf.summary.histogram('hidden_weights', weights['hidden_layer'])
        tf.summary.histogram('output_weights', weights['output'])
    
    with tf.name_scope('biases'):
        bias = {
            'hidden_layer':tf.Variable(tf.zeros([n_hidden1]), name='hidden_biases'),
            'output':tf.Variable(tf.zeros(n_labels), name='output_biases')
        }
        
        tf.summary.histogram('hidden_biases', bias['hidden_layer'])
        tf.summary.histogram('output_biases', bias['output'])
        
    with tf.name_scope('hidden_layers'):

        hidden_layer = tf.nn.bias_add(tf.matmul(inputs,weights['hidden_layer']), bias['hidden_layer'])
        hidden_layer = tf.nn.relu(hidden_layer, name='hidden_layer_output')
        
    with tf.name_scope('predictions'):

        logits = tf.nn.bias_add(tf.matmul(hidden_layer, weights['output']), bias['output'], name='logits')
        pred = tf.nn.softmax(logits, name='predictions')
        tf.summary.histogram('predictions', pred)
    
    with tf.name_scope('cost'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot, name='cross_entropy')
        cost = tf.reduce_mean(entropy, name='cost')
        tf.summary.scalar('cost', cost)
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        #tensorboard
        train_writer = tf.summary.FileWriter('./logs/3', sess.graph)
        
        for epoch in range(n_epochs):
            
            summary,_, loss = sess.run([merged,optimizer, cost], feed_dict={inputs:train_X, labels:train_Y})
           
            print("Epoch: {0} ; training loss: {1}".format(epoch, loss))
            
            train_writer.add_summary(summary, epoch+1)
            
        print('training finished')
        
         # testing the model on test data
            
#         test_loss,logits = sess.run([loss,logits],feed_dict={inputs:test_X,labels:test_Y})
#         predictions = tf.nn.softmax(logits)
        
#         correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf.one_hot(test_Y), 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) 
        
#         print('model accuracy : {}'.format(accuracy))
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({inputs: test_X, labels: test_Y}))
        
model3 = build_model_3()

