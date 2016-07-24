#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from init import *

# Model Hyperparameters
tf.flags.DEFINE_integer("num_layers", 1, "Number of filters per filter size (default: 150)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")

class LSTM_MODEL(object):

	def __init__(self, word_embeddings, batch_size, num_steps, num_classes, hidden_size, vocab_size, keep_prob, num_layers, l2_reg_lambda, is_training = True):
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		vocab_size = vocab_size
		size = hidden_size

		self.input_x = tf.placeholder(tf.int32, [None, num_steps], name = "input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(word_embeddings, name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
		
		inputs = self.embedded_chars
		if is_training and keep_prob < 1:
			inputs = tf.nn.dropout(inputs, keep_prob)

		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
		if is_training and keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell, output_keep_prob=keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
		self._initial_state = cell.zero_state(batch_size, tf.float32)

		outputs = []
		states = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
				states.append(state)

		self.lstm_last_h = tf.reshape(cell_output, [-1, size])


		l2_loss = tf.constant(0.0)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[size, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.lstm_last_h, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		print self.input_y.dtype
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
		lstm = LSTM_MODEL(
			word_embeddings = w, 
			batch_size = FLAGS.batch_size, 
			num_steps = len(x_train[0]), 
			num_classes = len(y_train[0]),
			hidden_size = len(w[0]), 
			vocab_size = len(w), 
			keep_prob = FLAGS.dropout_keep_prob,
			num_layers = FLAGS.num_layers,
			l2_reg_lambda = FLAGS.l2_reg_lambda)
		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(0.0005)
		grads_and_vars = optimizer.compute_gradients(lstm.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Initialize all variables
		sess.run(tf.initialize_all_variables())

		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
				lstm.input_x: x_batch,
				lstm.input_y: y_batch,
			}
			_, step, loss, accuracy = sess.run(
				[train_op, global_step, lstm.loss, lstm.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
 			if step % 100 == 0:
 				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

		def dev_step(x_batch, y_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
				lstm.input_x: x_batch,
				lstm.input_y: y_batch,
			}
			step, loss, accuracy = sess.run(
				[global_step, lstm.loss, lstm.accuracy], feed_dict)
			time_str = datetime.datetime.now().isoformat()
			# print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			return accuracy


		# Generate batches
		batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
		# Training loop. For each batch...
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
				# break

