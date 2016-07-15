#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from init import *

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 150)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# def load_data_and_labels():
# 	"""
# 	Loads MR polarity data from files, splits the data into words and generates labels.
# 	Returns split sentences and labels.
# 	"""
# 	print "Reading word embeddings......"
# 	word_embeddings = []
# 	f = open("vec.txt", "r")
# 	while True:
# 		content = f.readline()
# 		if content == "":
# 			break
# 		content = content.strip().split()
# 		content = [(float)(i) for i in content]
# 		word_embeddings.append(content)
# 	f.close()
# 	word_embeddings = np.array(word_embeddings, dtype=np.float32)

# 	print "Reading word vocabulary......"
# 	word2id = {}
# 	f = open("str.txt", "r")
# 	while True:
# 		content = f.readline()
# 		if content == "":
# 			break
# 		word2id[content.strip()] = len(word2id)
# 	f.close()

# 	print "Reading training file......"
# 	relationhash = {}
# 	x_train = []
# 	y_train = []
# 	f = open("train.txt", "r")
# 	content = f.readlines()
# 	for i in content:
# 		i = i.replace("#\n", "").split("\t")
# 		y_train.append(i[2])
# 		x_train.append(i[3])
# 		if not i[2] in relationhash:
# 			relationhash[i[2]] = len(relationhash)
# 	f.close()

# 	print "Reading testing file......"
# 	x_test = []
# 	y_test = []
# 	f = open("test.txt", "r")
# 	content = f.readlines()
# 	for i in content:
# 		i = i.replace("#\n", "").split("\t")
# 		y_test.append(i[2])
# 		x_test.append(i[3])
# 		if not i[2] in relationhash:
# 			relationhash[i[2]] = len(relationhash)
# 	f.close()

# 	res = []
# 	for i in xrange(0, len(y_test)):
# 		uid = relationhash[y_test[i]]
# 		label = [0 for i in range(0, len(relationhash))]
# 		label[uid] = 1
# 		res.append(label)
# 	y_test = np.array(res)

# 	res = []
# 	for i in xrange(0, len(y_train)):
# 		uid = relationhash[y_train[i]]
# 		label = [0 for i in range(0, len(relationhash))]
# 		label[uid] = 1
# 		res.append(label)
# 	y_train = np.array(res)

# 	max_document_length_train = max([len(x.split()) for x in x_train])
# 	max_document_length_test = max([len(x.split()) for x in x_test])
# 	max_document_length = max(max_document_length_train, max_document_length_test)

# 	size = len(x_train)
# 	for i in xrange(size):
# 		text = [0 for j in xrange(max_document_length)]
# 		content = x_train[i].split()
# 		for j in xrange(len(content)):
# 			if not content[j] in word2id:
# 				text[j] = 0
# 			else:
# 				text[j] = word2id[content[j]]
# 		x_train[i] = text
# 	x_train = np.array(x_train)

# 	size = len(x_test)
# 	for i in xrange(size):
# 		text = [0 for j in xrange(max_document_length)]
# 		content = x_test[i].split()
# 		for j in xrange(len(content)):
# 			if not content[j] in word2id:
# 				text[j] = 0
# 			else:
# 				text[j] = word2id[content[j]]
# 		x_test[i] = text
# 	x_test = np.array(x_test)

# 	return word_embeddings, x_train, y_train, x_test, y_test



# def batch_iter(data, batch_size, num_epochs, shuffle=True):
# 	"""
# 	Generates a batch iterator for a dataset.
# 	"""
# 	data = np.array(data)
# 	data_size = len(data)
# 	num_batches_per_epoch = int(len(data)/batch_size) 
# 	for epoch in range(num_epochs):
#  		# Shuffle the data at each epoch
# 		if shuffle:
# 			shuffle_indices = np.random.permutation(np.arange(data_size))
# 			shuffled_data = data[shuffle_indices]
# 		else:
# 			shuffled_data = data
# 		for batch_num in range(num_batches_per_epoch):
# 			start_index = batch_num * batch_size
# 			end_index = min((batch_num + 1) * batch_size, data_size)
# 			yield shuffled_data[start_index:end_index]

class TextCNN(object):
	def __init__(
		self, word_embeddings, sequence_length, num_classes, vocab_size,
		embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(word_embeddings, name="W")
			print W.get_shape()
			print W.dtype
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			print self.embedded_chars.get_shape()
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
			print self.embedded_chars_expanded.get_shape()

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				print W.get_shape()
				print W.dtype
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



w, x_train, y_train, x_dev, y_dev = load_data_and_labels()

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			word_embeddings = w,
			sequence_length=len(x_train[0]),
			num_classes=len(y_train[0]),
			vocab_size=len(w),
			embedding_size=len(w[0]),
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters=FLAGS.num_filters,
			l2_reg_lambda=FLAGS.l2_reg_lambda)
		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(0.001)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# # Keep track of gradient values and sparsity (optional)
		# grad_summaries = []
		# for g, v in grads_and_vars:
		# 	if g is not None:
		# 		grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
		# 		sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
		# 		grad_summaries.append(grad_hist_summary)
		# 		grad_summaries.append(sparsity_summary)
		# 		grad_summaries_merged = tf.merge_summary(grad_summaries)

		# # Output directory for models and summaries
		# timestamp = str(int(time.time()))
		# out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		# print("Writing to {}\n".format(out_dir))

		# # Summaries for loss and accuracy
		# loss_summary = tf.scalar_summary("loss", cnn.loss)
		# acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

		# # Train Summaries
		# train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
		# train_summary_dir = os.path.join(out_dir, "summaries", "train")
		# train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

		# # Dev summaries
		# dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
		# dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		# dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

		# # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		# checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		# checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		# if not os.path.exists(checkpoint_dir):
		# 	os.makedirs(checkpoint_dir)
		# saver = tf.train.Saver(tf.all_variables())

		# # Write vocabulary
		# vocab_processor.save(os.path.join(out_dir, "vocab"))

		# Initialize all variables
		sess.run(tf.initialize_all_variables())

		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
				cnn.input_x: x_batch,
				cnn.input_y: y_batch,
				cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
			}
			_, step, loss, accuracy = sess.run(
				[train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			if step % 100 == 0:
 				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			# train_summary_writer.add_summary(summaries, step)

		def dev_step(x_batch, y_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
				cnn.input_x: x_batch,
				cnn.input_y: y_batch,
				cnn.dropout_keep_prob: 1.0
			}
			step, loss, accuracy = sess.run(
				[global_step, cnn.loss, cnn.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			# print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			return accuracy
			# if writer:
			# 	writer.add_summary(summaries, step)

		# Generate batches
		batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
		# Training loop. For each batch...
		s = 0
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			train_step(x_batch, y_batch)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				res = 0.0
				tot = 0.0
				num = (int)(len(y_dev) / (float)(FLAGS.batch_size))
				print num
				for i in range(num):
					res = res + dev_step(x_dev[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], y_dev[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size])
					tot = tot + 1.0
				print res/tot

		print("\nEvaluation:")
		res = 0.0
		tot = 0.0
		num = (int)(len(y_dev) / (float)(FLAGS.batch_size))
		print num
		for i in range(num):
			res = res + dev_step(x_dev[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], y_dev[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size])
			tot = tot + 1.0
		print res/tot



				# print("")
			# if current_step % FLAGS.checkpoint_every == 0:
			# 	path = saver.save(sess, checkpoint_prefix, global_step=current_step)
			# 	print("Saved model checkpoint to {}\n".format(path))

