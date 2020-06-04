import _public as pb
import csv
import numpy as np
import random
# import os, shutil
# from gensim.models.word2vec import Word2Vec
# from nltk.corpus import wordnet as wn



class Example:
	def __init__(self):
		self.label = ""
		self.fgt_channels = []
		self.group = -1

		self.word2vec_mat = []
		self.char_mat = []
		self.bert_mat = []
		self.gpt2_mat = []
		self.transformerXL_mat = []
		self.elmo_mat = []
		self.glove_mat = []
		self.bow_vec = []
		self.rnn_mat = []
		self.mssm_vec_channels = []
		self.tag_mat = []

		self.mssm_probdis   = []
		self.mssm_label = ""
		self.mcts_label = ""

		self.tags = []

	def Print(self):
		print("%s(%d)\t%s" % (self.label, self.group, self.fgt_channels[0]))

def Preprocess_Text(raw_text):
	_raw_text, words = "", []
	for ch in raw_text:
		if ( ch not in pb.punctuations ):
			_raw_text += ch
		else:
			_raw_text += " " + ch + " "
	for word in _raw_text.split(" "):
		if (len(word) > 0):
			words.append(word)

	text = " ".join(words)
	text = text.strip()

	return text

def Preprocess_Array(raw_text):
	array = []

	raw_text = raw_text.replace("[", "")
	raw_text = raw_text.replace("]", "")
	raw_text = raw_text.replace(",", "")

	for value in raw_text.split(" "):
		if(len(value)>0):
			array.append( float(value) )

	return array

def Read_Data(path):

	examples_train, examples_test, group = [], [], 1

	examples = examples_train
	with open(path, encoding='UTF-8-sig') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			string = "\t".join(row)
			if("TEST" in string):
				examples = examples_test
				continue
			if(pb.Is_LabelText(string)):
				if(len(string.split("\t"))==2):
					raw_label, raw_text_l1 = string.split("\t")
					raw_text_l2 = ""
					raw_text_l3 = ""
					raw_text_l4 = ""
					raw_text_l5 = ""
					raw_text_l6 = ""
				elif(len(string.split("\t"))==6):
					raw_label, raw_text_l1, raw_text_l2, raw_text_l3, raw_text_l4, raw_text_l5 = string.split("\t")
				elif (len(string.split("\t"))==7):
					raw_label, raw_text_l1, raw_text_l2, raw_text_l3, raw_text_l4, raw_text_l5, raw_text_l6 = string.split("\t")

				example = Example()
				example.label = raw_label.upper()
				example.fgt_channels.append( Preprocess_Text(raw_text_l1) )
				example.fgt_channels.append( Preprocess_Text(raw_text_l2) )
				example.fgt_channels.append( Preprocess_Text(raw_text_l3) )
				example.fgt_channels.append( Preprocess_Text(raw_text_l4) )
				example.fgt_channels.append( Preprocess_Text(raw_text_l5) )
				# example.fgt_channels.append( Preprocess_Text(raw_text_l6) )
				example.group = group

				examples.append(example)

				if("." in example.fgt_channels[0]):
					group += 1

				for word in example.fgt_channels[0].split(" "):
					pb.vocabulary.add(word)
			else:
				group += 2

	return examples_train, examples_test

def Writer_Data(texts):
	pass
	# writer = open("./train_en.dat", "w")
	# pb.Print_Dotted_Line()
	# for text in texts_train:
	# 	for example in text:
	# 		writer.write(str(example.mssm_rep_en)+"\n")
	# 	writer.write("\n")
	# writer.close()
	#
	# writer = open("./train_ch.dat", "w")
	# for text in texts_train:
	# 	for example in text:
	# 		writer.write(str(example.mssm_rep_ch) + "\n")
	# 	writer.write("\n")
	# writer.close()
	#
	# writer = open("./test_en.dat", "w")
	# pb.Print_Dotted_Line()
	# for text in texts_test:
	# 	for example in text:
	# 		writer.write(str(example.mssm_rep_en) + "\n")
	# 	writer.write("\n")
	# writer.close()
	#
	# writer = open("./test_ch.dat", "w")
	# for text in texts_test:
	# 	for example in text:
	# 		writer.write(str(example.mssm_rep_ch) + "\n")
	# 	writer.write("\n")
	# writer.close()

def Init_All_Variables(examples):

	label_map = {}
	for example in examples:
		if (example.label not in label_map.keys()):
			label_map[example.label] = 1
		else:
			label_map[example.label] = label_map[example.label] + 1

	pb.label_histogram_x, pb.label_histogram_y = pb.Map_To_Sorted_List(label_map)

	print("The Number of Labels (Training Examples): %d: %s %s" % (len(pb.label_histogram_x), str(pb.label_histogram_x), str(pb.label_histogram_y)))

def Get_Example_Coding(example):
	# x = [example.bert_mat]

	x = []
	for layer in example.bert_mat:
		if (len(x) == 0):
			x = np.array(layer)
		else:
			x += np.array(layer)

	# x = []
	# for layer in example.bert_mat:
	# 	if (len(x) == 0):
	# 		x = np.array(layer)
	# 		# print(len(layer))
	# 	else:
	# 		x += np.array(layer)
	# x = [[x]]

	v1 = np.zeros(1)
	v2 = np.zeros(1)
	v3 = np.zeros(1)
	v4 = np.zeros(1)
	v5 = np.zeros(1)
	v6 = np.zeros(1)

	y = pb.label_histogram_x.index(example.label)

	return x, v1, v2, v3, v4, v5, v6, y

	# if(len(pb.pos_sorted)>0):
	# 	x_pos = []
	# 	for pos in example.tag_sentence.split("_"):
	# 		x_pos.append(pb.pos_sorted.index(pos))
	# 	while(len(x_pos)<pb.sentence_max_len):
	# 		x_pos.extend([0.0])
	# 	x.extend(x_pos)
	#
	# x_synset = []
	# for word in example.sentence.split("_"):
	# 	if(pb.Is_Word(word)):
	# 		verb_len   = len(wn.synsets(word, pos=wn.VERB))
	# 		noun_len   = len(wn.synsets(word, pos=wn.NOUN))
	# 		adj_len    = len(wn.synsets(word, pos=wn.ADJ))
	# 		adjsat_len = len(wn.synsets(word, pos=wn.ADJ_SAT))
	# 		adv_len    = len(wn.synsets(word, pos=wn.ADV))
	# 		x_synset.extend( [verb_len, noun_len, adj_len, adjsat_len, adv_len] )
	# while (len(x_synset) < pb.sentence_max_len*5):
	# 	x_synset.extend([0.0])
	# x.extend(x_synset)
	#
	# x_wfd = []
	# for word in example.sentence.split("_"):
	# 	if(word in pb.word_freq_map.keys()):
	# 		x_wfd.extend( pb.word_freq_map[word] )
	# while (len(x_wfd) < pb.sentence_max_len):
	# 	x_wfd.extend([0.0])
	# x.extend(x_wfd)

	# 添加噪声
	# noise_weight = 0.007
	# for i in range(len(x)):
	# 	x[i] += noise_weight * random.random()

def Get_Balanced_Data(examples):

	examples_list = [[] for _ in pb.label_histogram_x]
	sampled_examples_list = [[] for _ in pb.label_histogram_x]

	for example in examples:
		label = example.label
		index = pb.label_histogram_x.index(label)
		examples_list[index].append(example)
		sampled_examples_list[index].append(example)

	sampled_examples, sample_num = [], np.max([len(obj) for obj in examples_list])

	for i in range(len(sampled_examples_list)):
		while (len(sampled_examples_list[i]) < sample_num):
			example = random.choice( examples_list[i] )
			sampled_examples_list[i].append(example)
		sampled_examples.extend(sampled_examples_list[i])

	return sampled_examples

def Get_Encoded_Data(examples):

	xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = [], [], [], [], [], [], [], []
	for example in examples:
		x, v1, v2, v3, v4, v5, v6, y = Get_Example_Coding(example)
		xs.append(x)
		v1s.append(v1)
		v2s.append(v2)
		v3s.append(v3)
		v4s.append(v4)
		v5s.append(v5)
		v6s.append(v6)
		ys.append(y)

	return xs, v1s, v2s, v3s, v4s, v5s, v6s, ys

def Print_Report(examples):
	up_mssm, up_mcts = 0, 0
	for example in examples:
		if(example.mssm_label==example.label):
			up_mssm += 1
		if(example.mcts_label== example.label):
			up_mcts += 1

	print("MSSM Accuracy: {:.2%}".format(up_mssm * 1.0 / len(examples)))
	print("MCTS Accuracy: {:.2%}".format(up_mcts * 1.0 / len(examples)))

def Print_Examples(examples):
	for example in examples:
		example.Print()
