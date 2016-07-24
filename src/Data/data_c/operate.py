#coding:utf-8


import numpy
import random

hash = {}

f = open("pre_weibo30w_limit50.txt", "r")
content = f.readlines()
for i in content:
	# print i
	i = i.strip().replace("　","").split()
	lists = ""
	label = None
	position = None
	s = 0
	for j in i:
		if j.find("emo_wb_") != -1:
			label = j
			if position == None:
				position = s
		else:
			lists = lists + " " + j
		s = s + 1
	lists = lists.strip()
	if lists == "":
		continue
	if not label in hash:
		hash[label] = []
	hash[label].append((lists, position))
f.close()

train = open("train.txt", "w")
test = open("test.txt", "w")
for i in hash:
	label = i
	for j in hash[i]:
		sentence = j[0]
		position = j[1]
		if random.random() >= 0.8:
			test.write("%s\t%s\t%d\n"%(sentence,label,position))
		else:
			train.write("%s\t%s\t%d\n"%(sentence,label,position))
test.close()
train.close()
