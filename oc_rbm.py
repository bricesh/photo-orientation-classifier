import tensorflow as tf
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def map_class(clab):
    img_class = [0,0,0,0]
    if clab == "u":
        img_class = [1,0,0,0]
    if clab == "d":
        img_class = [0,1,0,0]
    if clab == "l":
        img_class = [0,0,1,0]
    if clab == "r":
        img_class = [0,0,0,1]
    return img_class

PIK = 'pickle.dat'
with open(PIK, "rb") as f:
    img_data = pickle.load(f, encoding="latin1")

img_label = img_data[0]
img_label = img_label.file_orientation.apply(lambda x: map_class(x))
img_label = np.vstack(img_label)
img_flat_data = img_data[1]


n_visible = img_flat_data[1].size
n_hidden = 40000
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


X_train, X_test, y_train, y_test = train_test_split(img_flat_data,
                                                    img_label,
                                                    test_size=.5,
                                                    random_state=42)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

n_nodes_hl1 = 5000
n_nodes_hl2 = 2000
n_nodes_hl3 = 500

n_classes = 4
batch_size = 100
hm_epochs = 70

x = tf.placeholder('float', [None, 59658])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([59658, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(X_train):
				start = i
				end = i+batch_size
				batch_x = np.array(X_train[start:end])
				batch_y = np.array(y_train[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size

			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:X_test, y:y_test}))

train_neural_network(x)



#
#
#
#
#url = "https://www.dropbox.com/[something]/[filename]?dl=1"  # dl=1 is important
#import urllib.request
#u = urllib.request.urlopen(url)
#data = u.read()
#u.close()
 
#with open([filename], "wb") as f :
#    f.write(data)
#
#
#
#
#
#
