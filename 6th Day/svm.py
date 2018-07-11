import tensorflow as tf
import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
import plot_boundary_on_data  

BATCH_SIZE = 100  


tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of training epochs.')
tf.app.flags.DEFINE_float('svmC', 1,
                            'The C parameter of the SVM cost function.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')
FLAGS = tf.app.flags.FLAGS


def extract_data(filename):

    out = np.loadtxt(filename, delimiter=',');

    labels = out[:,0]
    labels = labels.reshape(labels.size,1)
    fvecs = out[:,1:]

    return fvecs,labels


def main(argv=None):
    verbose = FLAGS.verbose

    plot = FLAGS.plot
    
    train_data_filename = FLAGS.train
    train_data,train_labels = extract_data(train_data_filename)
    train_labels[train_labels==0] = -1
    train_size,num_features = train_data.shape
    num_epochs = FLAGS.num_epochs
    svmC = FLAGS.svmC
    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None,1])

   
    W = tf.Variable(tf.zeros([num_features,1]))
    b = tf.Variable(tf.zeros([1]))
    y_raw = tf.matmul(x,W) + b

    # Optimization.
    regularization_loss = 0.5*tf.reduce_sum(tf.square(W)) 
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 
        1 - y*y_raw));
    svm_loss = regularization_loss + svmC*hinge_loss;
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    # Evaluation.
    predicted_class = tf.sign(y_raw);
    correct_prediction = tf.equal(y,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as s:
        tf.initialize_all_variables().run()
        if verbose:
            print ('Initialized!')
            print ()
            print ('Training.')

        
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print (step),
                
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y: batch_labels})
            print ('loss: ', svm_loss.eval(feed_dict={x: batch_data, y: batch_labels}))
            
            if verbose and offset >= train_size-BATCH_SIZE:
                print


        if verbose:
            print
            print ('Weight matrix.')
            print (s.run(W))
            print
            print ('Bias vector.')
            print (s.run(b))
            print
            print ("Applying model to first test instance.")
            print
            
        print ("Accuracy on train:", accuracy.eval(feed_dict={x: train_data, y: train_labels}))

        if plot:
            eval_fun = lambda X: predicted_class.eval(feed_dict={x:X}); 
            plot_boundary_on_data.plot(train_data, train_labels, eval_fun)
    
if __name__ == '__main__':
    tf.app.run()
