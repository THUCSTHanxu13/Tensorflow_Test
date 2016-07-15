#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime

def load_data_and_labels():
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	print "Reading word embeddings......"
	word_embeddings = []
	word2id = {}
	f = open("data/pre_weibo30w.txt.100.vec", "r")
	content = f.readline()
	while True:
		content = f.readline()
		if content == "":
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		word_embeddings.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	lists = [0.0 for i in range(len(word_embeddings[0]))]
	word_embeddings.append(lists)
	word_embeddings.append(lists)
	word_embeddings = np.array(word_embeddings, dtype=np.float32)

	print "Reading training file......"
	relationhash = {}
	x_train = []
	y_train = []
	f = open("data/train.txt", "r")
	content = f.readlines()
	for i in content:
		i = i.split("\t")
		y_train.append(i[1])
		x_train.append(i[0])
		if not i[1] in relationhash:
			relationhash[i[1]] = len(relationhash)
	f.close()

	print "Reading testing file......"
	x_test = []
	y_test = []
	f = open("data/test.txt", "r")
	content = f.readlines()
	for i in content:
		i = i.split("\t")
		y_test.append(i[1])
		x_test.append(i[0])
		if not i[1] in relationhash:
			relationhash[i[1]] = len(relationhash)
	f.close()

	res = []
	for i in xrange(0, len(y_test)):
		uid = relationhash[y_test[i]]
		label = [0 for i in range(0, len(relationhash))]
		label[uid] = 1
		res.append(label)
	y_test = np.array(res)

	res = []
	for i in xrange(0, len(y_train)):
		uid = relationhash[y_train[i]]
		label = [0 for i in range(0, len(relationhash))]
		label[uid] = 1
		res.append(label)
	y_train = np.array(res)

	for x in x_train:
		if len(x.strip().split()) >50:
			print x

	for x in x_test:
		if len(x.strip().split()) >50:
			print x
	max_document_length_train = max([len(x.split()) for x in x_train])
	max_document_length_test = max([len(x.split()) for x in x_test])
	max_document_length = max(max_document_length_train, max_document_length_test)

	size = len(x_train)
	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_train[i].split()
		for j in xrange(len(content)):
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		x_train[i] = text
	x_train = np.array(x_train)

	size = len(x_test)
	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_test[i].split()
		for j in xrange(len(content)):
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		x_test[i] = text
	x_test = np.array(x_test)

	return word_embeddings, x_train, y_train, x_test, y_test

# word_embeddings, x_train, y_train, x_test, y_test =  load_data_and_labels()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	print "hx"
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = (int)(round(len(data)/batch_size)) 
	print num_batches_per_epoch
	for epoch in range(num_epochs):
 		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]