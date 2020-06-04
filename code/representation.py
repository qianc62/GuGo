#!/usr/bin/env Python
# coding=utf-8


import _public as pb
from scipy import optimize
import math
import numpy as np
import plot as fig
from gensim.models.word2vec import Word2Vec
import dataset
import torch
import os
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel
from allennlp.commands.elmo import ElmoEmbedder
import random
# from allennlp.commands.elmo import ElmoEmbedder
# from scipy import stats
# import matplotlib.pyplot as plt
# from scipy import stats
# import dataset as ds
# import random as random
# import threading
# import time
import pickle



class Seed:
	def __init__(self):
		self.word = ""
		self.chi = 0.0
		self.ml, self.sl, self.al, self.hl = [], [], [], []

class MSSM:
	def __init__(self,texts):
		self.cell_perception_window_size = 6
		self.char_histogram_channels = []
		self.char_histogram_channels = []
		self.cell_map_channels = {}
		self.seeds_channels = []

		# self.char_histogram_x_en, self.char_histogram_y_en = self.Get_Char_Histogram(fgts_en)
		# self.char_histogram_x_ch, self.char_histogram_y_ch = self.Get_Char_Histogram(fgts_ch)
		# self.cell_map_en = self.Get_Cells(self.char_histogram_x_en, self.char_histogram_y_en, self.cell_perception_window_size_en)
		# self.cell_map_ch = self.Get_Cells(self.char_histogram_x_ch, self.char_histogram_y_ch, self.cell_perception_window_size_ch)
		# fig.Plot_Cells(self)

		if(os.path.exists("./data/"+pb.dataset+"_seeds[4].dat")==False):
			seed_minlength = [2, 1, 2, 2, 1]
			seed_maxlength = [pb.word_maxlength, 5, pb.word_maxlength, pb.word_maxlength, 5]

			labels = [example.label for example in texts]
			for i in range(pb.language_channel_num):
				if(i<=4):
					continue

				ftgs = [example.fgt_channels[i] for example in texts]
				seed_candidates = self.Seed_Candidate_Generation(ftgs, seed_minlength[i], seed_maxlength[i])
				print("seed_candidates:{}".format(len(seed_candidates)))
				seeds = self.Seed_Generation(ftgs, labels, seed_candidates)
				print("seeds:{}".format(len(seeds)))
				self.seeds_channels.append(seeds)

				writer = open("./data/" + pb.dataset + "_seeds[" + str(i) + "].dat", "w")
				for seed in seeds:
					writer.write("{}\n".format(seed))
				writer.close()
				print("./data/" + pb.dataset + "_seeds[" + str(i) + "].dat")
		else:
			for i in range(pb.language_channel_num):
				seeds = open("./data/"+pb.dataset+"_seeds["+str(i)+"].dat").read().split("\n")
				if(i==0 or i==2 or i==3):
					self.seeds_channels.append([seed for seed in seeds if len(seed)>1])
				else:
					self.seeds_channels.append(seeds)

		pb.seeds_channels = self.seeds_channels

	# def Get_Char_Histogram(self, fgts):
	# 	char_map = {}
	# 	for fgt in fgts:
	# 		for char in fgt:
	# 			if (char not in char_map.keys()):
	# 				char_map[char] = 1
	# 			else:
	# 				char_map[char] = char_map[char] + 1
	# 	return pb.Map_To_Sorted_List(char_map)
	#
	# def Curve_Fit(self,y):
	# 	x = []
	# 	for i in range(len(y)):
	# 		x.append(i)
	#
	# 	y = np.array(y)
	# 	y = y / np.max(y)
	#
	# 	mu, sigma, a, h = optimize.curve_fit(self.Gauss, x, y, bounds=(0, [len(y), len(y), 1.0, 1.0]))[0]
	#
	# 	return mu, sigma, a, h
	#
	# def Get_Cells(self, char_histogram_x, char_histogram_y, cell_perception_window_size):
	# 	cell_map = {}
	#
	# 	normalized_factor = np.max(char_histogram_y)
	#
	# 	for i in range(len(char_histogram_x)):
	# 		print(char_histogram_x[i])
	# 		y = []
	# 		for j in range(len(char_histogram_x)):
	# 			value = math.fabs(char_histogram_y[i] - char_histogram_y[j])
	# 			value = value / normalized_factor
	# 			value = math.exp(-value)
	# 			y.append(value)
	#
	# 		left_index  = i - cell_perception_window_size
	# 		right_index = i + cell_perception_window_size
	# 		if (left_index < 0):
	# 			left_index = 0
	# 		if (right_index >= len(char_histogram_x)):
	# 			right_index = len(char_histogram_x) - 1
	#
	# 		wy = y[left_index:right_index + 1]
	# 		wy = np.array(wy)
	# 		wy = np.exp(wy) / np.sum(np.exp(wy))
	# 		wy = (wy - np.min(wy)) / (np.max(wy) - np.min(wy))
	#
	# 		ans = []
	# 		for i in range(len(y)):
	# 			if (i < left_index or i > right_index):
	# 				ans.append(0.0)
	# 			else:
	# 				ans.append(wy[i - left_index])
	#
	# 		mu, sigma, a, h = self.Curve_Fit(ans)
	#
	# 		cell_map[char_histogram_x[i]] = (ans, mu, sigma, a, h)
	#
	# 	return cell_map

	def Seed_Candidate_Generation(self, fgts, seed_minlength, seed_maxlength):
		seed_candidates = set()
		for fgt in fgts:
			for begin_index in range(len(fgt)):
				for l in range(seed_minlength, seed_maxlength + 1):
					if (begin_index + l - 1 <= len(fgt) - 1):
						gram = fgt[begin_index:begin_index + l]
						for ch in gram:
							if(ch in pb.punctuations):
								break
						else:
							seed_candidates.add(gram)
		return seed_candidates

	def Chi_Square_Test(self, fgts, labels, seed_candidate):

		u = np.zeros( len(pb.label_histogram_x) )
		v = np.zeros( len(pb.label_histogram_x) )

		for i in range(len(fgts)):
			if (seed_candidate in fgts[i]):
				u[pb.label_histogram_x.index(labels[i])] += 1
			else:
				v[pb.label_histogram_x.index(labels[i])] += 1

		sum_u = np.sum(u)
		sum_v = np.sum(v)
		ratio_u = sum_u * 1.0 / (sum_u + sum_v)
		ratio_v = sum_v * 1.0 / (sum_u + sum_v)

		chi = 0.0
		for i in range(len(pb.label_histogram_x)):
			e_u = (u[i] + v[i]) * ratio_u
			e_v = (u[i] + v[i]) * ratio_v
			chi += (u[i] - e_u) ** 2 / (e_u + 0.00000001)
			chi += (v[i] - e_v) ** 2 / (e_v + 0.00000001)

		return chi

	# def Get_Mixture_Gauss_Para(self,word):
	# 	ml, sl, al, hl = [], [], [], []
	# 	for char in word:
	# 		tuple = self.cell_map_en[char]
	# 		ml.append(tuple[1])
	# 		sl.append(tuple[2])
	# 		al.append(tuple[3])
	# 		hl.append(tuple[4])
	# 	# fig.Plot_Mixture_Gauss(word, ml, sl, al, hl)
	# 	return ml, sl, al, hl

	def Seed_Generation(self, fgts, labels, seed_candidates):
		seeds = []

		chi_map = {}
		for i,seed_candidate in enumerate(seed_candidates):
			chi = self.Chi_Square_Test(fgts, labels, seed_candidate)
			chi_map[seed_candidate] = chi
			print("{:.2%}".format(i*1.0/len(seed_candidates)))

		chi_sorted_x, chi_sorted_y = pb.Map_To_Sorted_List(chi_map)

		flag = [True for _ in chi_sorted_x]
		for i in range(len(chi_sorted_x)):
			if(flag[i]==False):
				continue
			for j in range(i + 1, len(chi_sorted_x)):
				if (chi_sorted_x[i] in chi_sorted_x[j]):
					flag[j] = False
			print("{:.2%}".format(i*1.0/len(chi_sorted_x)))

		for i in range(len(chi_sorted_x)):
			if (flag[i] == True and chi_sorted_y[i]>0.00):
				seed = Seed()
				seed.word = chi_sorted_x[i]
				seed.chi = chi_map[seed.word]
				seeds.append(seed.word)
				# print(seed.word)

		return seeds

	def Get_MSSM_Representation(self, texts):
		for example in texts:
			for i in range(len(example.fgt_channels)):
				representation = []
				for k in range(len(self.seeds_channels[i])):
					value = 1.0 if self.seeds_channels[i][k] in example.fgt_channels[i] else 0.0
					representation.append(value)
				example.mssm_vec_channels.append(representation)

				# x = np.arange(0, len(pb.char_sorted_x), 0.1)
				# y1 = self.Mixture_Gauss(x, ml1, sl1, al1, hl1)
				# y2 = sself.Mixture_Gauss(x, ml2, sl2, al2, hl2)
				# y1 = np.array(y1) + 0.0001
				# y2 = np.array(y2) + 0.0001
				# kl = stats.entropy(y1, y2)
				# return 1.0 - kl

	# def Gauss(self, x, mu, sigma, a, h ):
	# 	return a * np.exp( -(x-mu)**2 / (2*sigma**2) ) + h

class TextEmbedding:
	def __init__(self):
		# self.w2v_embdding_size = 100
		# self.w2v = Word2Vec.load("./w2v/w2v_model")
		# self.vocabulary = set(open("./w2v/text8_vocabulary.txt").read().split("\n"))
		# self.default_word = "a"

		self.bert_tokenizer = None
		self.bert = None

		# self.gpt2_tokenizer = None
		# self.gpt2 = None
        #
		# self.transformer_tokenizer = None
		# self.transformer = None
        #
		# self.elmo = None
        #
		# self.bert_map = {}

	def Get_Word2Vec_Representation(self, examples):
		for example in examples:
			representation = []
			for word in example.fgt_channels[0].split(" "):
				if(word in self.vocabulary):
					representation.append(self.w2v[word])
				else:
					representation.append(self.w2v[self.default_word])
			while (len(representation) < pb.fgt_maxlength):
				representation.append( np.zeros(self.w2v_embdding_size) )
			example.word2vec_mat = representation[0:pb.fgt_maxlength]

	def Get_RNN_Representation(self, examples):
		for example in examples:
			representation = []
			for word in example.fgt_channels[0].split(" "):
				if(word in self.vocabulary):
					representation.append(self.w2v[word])
				else:
					representation.append(self.w2v[self.default_word])
			while (len(representation) < pb.fgt_maxlength):
				representation.append( np.zeros(self.w2v_embdding_size) )
			example.rnn_mat = representation[0:pb.fgt_maxlength]

	def Get_Char_Representation(self, examples):
		for example in examples:
			representation = []
			for char in example.fgt_channels[0]:
				if(char in self.vocabulary):
					representation.append(self.w2v[char])
				else:
					_rep = [0.0 for _ in range(len(self.w2v[self.default_word]))]
					if( char in pb.label_histogram_x ):
						_rep[pb.label_histogram_x.index(char)] = 1
					representation.append(_rep)
			while (len(representation) < pb.fgt_maxlength*pb.word_maxlength):
				representation.append( np.zeros(self.w2v_embdding_size) )
			example.char_mat = representation[0:pb.fgt_maxlength*pb.word_maxlength]

	def Get_Bert_Representation(self, examples_train, examples_test):

		train_rep_file = "./data/"+pb.dataset+"_train_"+"bert"
		test_rep_file  = "./data/"+pb.dataset+"_test_"+"bert"

		if (os.path.exists(train_rep_file)==True and os.path.exists(test_rep_file)==True):
			with open(train_rep_file, 'rb') as file:
				examples_train_rep = pickle.load(file)
				for i, example in enumerate(examples_train):
					example.bert_mat = examples_train_rep[i]
			with open(test_rep_file, 'rb') as file:
				examples_test_rep = pickle.load(file)
				for i, example in enumerate(examples_test):
					example.bert_mat = examples_test_rep[i]
		else:
			examples = []
			for example in examples_train:
				examples.append(example)
			for example in examples_test:
				examples.append(example)

			for i, example in enumerate(examples):

				if(self.bert_tokenizer==None):
					self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

				text = "[CLS] " + example.fgt_channels[0] + " [SEP]"
				text = text.replace("  ", " ")
				tokenized_text = self.bert_tokenizer.tokenize(text)

				indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
				segments_ids = [0 for _ in tokenized_text]

				tokens_tensor = torch.tensor([indexed_tokens])
				segments_tensors = torch.tensor([segments_ids])

				if (self.bert == None):
					self.bert = BertModel.from_pretrained('bert-base-uncased')
					self.bert.eval()

				with torch.no_grad():
					representation, sum = [], 0

					encoded_layers, _ = self.bert(tokens_tensor, segments_tensors)
					a, b  = encoded_layers[0].numpy().shape[1], encoded_layers[0].numpy().shape[2]
					representation = np.zeros((a,b))

					for layer in encoded_layers:
						for words in layer.numpy():
							representation += words
							sum += 1
					if(sum>0):
						representation = representation * 1.0 / sum

					representation = list(representation)
					while (len(representation) < pb.fgt_maxlength):
						representation.append(np.zeros(b))

					example.bert_mat = representation[0:pb.fgt_maxlength]

				print("{:.2%}".format(i * 1.0 / len(examples)))

	def _Get_Bert_Representation(self):

		count = 0
		bert_map = {}
		for root, dirs, files in os.walk("./data/test"):
			for file in files:
				file_path = os.path.join(root, file)
				print(file_path)

				file = open(file_path, "r")
				while True:
					line = file.readline()
					if not line:
						break
					line = line[:len(line) - 1]
					line = line.split(" ")
					line = line[:len(line)-1]
					line = " ".join(line)

					if(line in bert_map.keys()):
						continue

					if(self.bert_tokenizer==None):
						self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

					text = "[CLS] " + line + " [SEP]"
					text = text.replace("  ", " ")
					tokenized_text = self.bert_tokenizer.tokenize(text)

					indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
					segments_ids = [0 for _ in tokenized_text]

					tokens_tensor = torch.tensor([indexed_tokens])
					segments_tensors = torch.tensor([segments_ids])

					if (self.bert == None):
						self.bert = BertModel.from_pretrained('bert-base-uncased')
						self.bert.eval()

					with torch.no_grad():
						representation, sum = [], 0

						encoded_layers, _ = self.bert(tokens_tensor, segments_tensors)

						Len = len(encoded_layers[-1].numpy()[0])
						representation = np.zeros(768)
						for i in range(1,Len-1):
							representation += encoded_layers[-1].numpy()[0][i]
							sum += 1
						representation = representation * 1.0 / sum

						bert_map[line] = representation

						count += 1
						if(count%100==0):
							print(count)

		with open("./bert_map", 'wb') as file:
			pickle.dump(bert_map, file)

	def _Get_Word_Bert_Representation(self, word):

		if(word not in self.bert_map.keys()):
			if(self.bert_tokenizer==None):
				self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

			text = "[CLS] " + word + " [SEP]"
			text = text.replace("  ", " ")
			tokenized_text = self.bert_tokenizer.tokenize(text)

			indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
			segments_ids = [0 for _ in tokenized_text]

			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			if (self.bert == None):
				self.bert = BertModel.from_pretrained('bert-base-uncased')
				self.bert.eval()

			with torch.no_grad():
				representation, sum = [], 0

				encoded_layers, _ = self.bert(tokens_tensor, segments_tensors)

				Len = len(encoded_layers[-1].numpy()[0])
				representation = np.zeros(768)
				for i in range(1,Len-1):
					representation += encoded_layers[-1].numpy()[0][i]
					sum += 1
				representation = representation * 1.0 / sum

				self.bert_map[word] = representation

		return self.bert_map[word]

	def Get_BOW_Representation(self, examples):
		volist = list(pb.vocabulary)
		for example in examples:
			x = [0.0 for _ in volist]
			for word in example.fgt_channels[0].split(" "):
				if (word in volist):
					index = volist.index(word)
					x[index] += 1
			example.bow_vec = x

	def Get_GPT2_Representation(self, examples):
		for i, example in enumerate(examples):

			# example.gpt2_mat = np.zeros((pb.fgt_maxlength,768))
			# continue

			if(self.gpt2_tokenizer==None):
				self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

			text = example.fgt_channels[0]
			indexed_tokens = self.gpt2_tokenizer.encode(text)
			tokens_tensor = torch.tensor([indexed_tokens])

			if (self.gpt2 == None):
				self.gpt2 = GPT2Model.from_pretrained('gpt2')
				self.gpt2.eval()

			with torch.no_grad():
				hidden_states, past = self.gpt2(tokens_tensor) # (1, 5, 768)
				shape = np.array(hidden_states).shape

				representation, sum = [], 0

				a, b  = shape[1], shape[2]
				representation = np.zeros((a,b))

				for layer in hidden_states:
					for words in layer.numpy():
						representation += words
						sum += 1
				if(sum>0):
					representation = representation * 1.0 / sum

				representation = list(representation)
				while (len(representation) < pb.fgt_maxlength):
					representation.append(np.zeros(b))

				example.gpt2_mat = representation[0:pb.fgt_maxlength]

			print("{:.2%}".format(i * 1.0 / len(examples)))

	def Get_Transformer_Representation(self, examples_train, examples_test):

		train_rep_file = "./data/" + pb.dataset + "_train_" + "transformerXL"
		test_rep_file = "./data/" + pb.dataset + "_test_" + "transformerXL"

		if (os.path.exists(train_rep_file) == True and os.path.exists(test_rep_file) == True):
			with open(train_rep_file, 'rb') as file:
				examples_train_rep = pickle.load(file)
				for i, example in enumerate(examples_train):
					example.transformerXL_mat = examples_train_rep[i]
			with open(test_rep_file, 'rb') as file:
				examples_test_rep = pickle.load(file)
				for i, example in enumerate(examples_test):
					example.transformerXL_mat = examples_test_rep[i]
		else:
			examples = []
			for example in examples_train:
				examples.append(example)
			for example in examples_test:
				examples.append(example)

			for i, example in enumerate(examples):

				# example.transformerXL_mat = np.zeros((pb.fgt_maxlength,20))
				# continue

				if(self.transformer_tokenizer==None):
					self.transformer_tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

				text = example.fgt_channels[0]
				tokenized_text = self.transformer_tokenizer.tokenize(text)

				indexed_tokens = self.transformer_tokenizer.convert_tokens_to_ids(tokenized_text)

				tokens_tensor = torch.tensor([indexed_tokens])

				if (self.transformer == None):
					self.transformer = TransfoXLModel.from_pretrained('transfo-xl-wt103')
					self.transformer.eval()

				with torch.no_grad():
					hidden_states, _ = self.transformer(tokens_tensor) # (1, 3, 1024)
					shape = np.array(hidden_states).shape
					# print(shape)

					representation, sum = [], 0

					a, b  = shape[1], shape[2]
					representation = np.zeros((a,b))

					for layer in hidden_states:
						for words in layer.numpy():
							representation += words
							sum += 1
					if(sum>0):
						representation = representation * 1.0 / sum

					representation = list(representation)
					while (len(representation) < pb.fgt_maxlength):
						representation.append(np.zeros(b))

					example.transformerXL_mat = representation[0:pb.fgt_maxlength]

				print("{:.2%}".format(i*1.0/len(examples)))

	def Get_ELMo_Representation(self, examples):
		for i, example in enumerate(examples):

			# example.elmo_mat = np.zeros((pb.fgt_maxlength,*))
			# continue

			if(self.elmo==None):
				options_file = "./sources/elmo_2x1024_128_2048cnn_1xhighway_options.json"
				weight_file = "./sources/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
				self.elmo = ElmoEmbedder(options_file, weight_file)

			text = example.fgt_channels[0]

			context_tokens = [text.split(" ")]
			elmo_embedding, _ = self.elmo.batch_to_embeddings(context_tokens)

			shape = np.array(elmo_embedding[0]).shape
			# print(shape)

			representation, sum = [], 0

			a, b  = shape[1], shape[2]
			representation = np.zeros((a,b))

			for layer in elmo_embedding:
				for words in layer.numpy():
					representation += words
					sum += 1
			if(sum>0):
				representation = representation * 1.0 / sum

			representation = list(representation)
			while (len(representation) < pb.fgt_maxlength):
				representation.append(np.zeros(b))

			example.elmo_mat = representation[0:pb.fgt_maxlength]

			print("{:.2%}".format(i*1.0/len(examples)))

	def Get_Glove_Representation(self, examples):
		glove_vocabulary = {}
		Len = 300

		for line in open("./sources/glove_"+str(Len)+"d.dat").read().split("\n"):
			eles = line.split(" ")
			if(len(eles)==Len+1):
				word = eles[0]
				vector = [float(ele) for ele in eles[1:]]
				glove_vocabulary[word] = vector

		for example in examples:
			representation = []
			for word in example.fgt_channels[0].split(" "):
				if (word in glove_vocabulary.keys()):
					representation.append(glove_vocabulary[word])
				else:
					representation.append(glove_vocabulary[self.default_word])
			while (len(representation) < pb.fgt_maxlength):
				representation.append(np.zeros(Len))
			example.glove_mat = representation[0:pb.fgt_maxlength]
			# print(np.array(example.glove_mat).shape)

	def _Get_Glove_Representation(self, examples):
		glove_vocabulary = {}
		Len = 300

		for line in open("./sources/glove_"+str(Len)+"d.dat").read().split("\n"):
			eles = line.split(" ")
			if(len(eles)==Len+1):
				word = eles[0]
				vector = [float(ele) for ele in eles[1:]]
				glove_vocabulary[word] = vector

		for example in examples:
			representation = []
			for word in example.fgt_channels[0].split(" "):
				if (word in glove_vocabulary.keys()):
					representation.append(glove_vocabulary[word])
				else:
					representation.append(glove_vocabulary[self.default_word])
			while (len(representation) < pb.fgt_maxlength):
				representation.append(np.zeros(Len))
			example.glove_mat = representation[0:pb.fgt_maxlength]
			# print(np.array(example.glove_mat).shape)


	def Get_Tag_Representation(self, examples_train, examples_test):
		examples = []
		for example in examples_train:
			examples.append(example)
		for example in examples_test:
			examples.append(example)

		tag_vocabulary = set()

		for i, example in enumerate(examples):
			tags = pb.Get_POS(example.fgt_channels[0])
			example.tags = tags
			for tag in tags:
				tag_vocabulary.add(tag)

		tag_vocabulary = sorted(list(tag_vocabulary))

		for example in examples:
			representation = []
			for tag in example.tags:
				x = np.zeros(len(tag_vocabulary))
				index = tag_vocabulary.index(tag)
				x[index] = 1.0
				representation.append(x)
			while (len(representation) < pb.fgt_maxlength):
				representation.append(np.zeros(len(tag_vocabulary)))
			example.tag_mat = representation[0:pb.fgt_maxlength]

	def Get_Bert_Representation_Tmp(self, sentences):

		sentences_bert = []

		for i in range(len(sentences)):
			print(sentences[i])

			if (self.bert_tokenizer == None):
				self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

			text = "[CLS] " + sentences[i] + " [SEP]"
			text = text.replace("  ", " ")
			tokenized_text = self.bert_tokenizer.tokenize(text)

			indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
			segments_ids = [0 for _ in tokenized_text]

			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			if (self.bert == None):
				self.bert = BertModel.from_pretrained('bert-base-uncased')
				self.bert.eval()

			with torch.no_grad():
				representation, sum = [], 0

				encoded_layers, _ = self.bert(tokens_tensor, segments_tensors)
				a, b = encoded_layers[0].numpy().shape[1], encoded_layers[0].numpy().shape[2]
				representation = np.zeros((a, b))

				for layer in encoded_layers:
					for words in layer.numpy():
						representation += words
						sum += 1
				if (sum > 0):
					representation = representation * 1.0 / sum

				representation = list(representation)
				while (len(representation) < pb.fgt_maxlength):
					representation.append(np.zeros(b))

				example.bert_mat = representation[0:pb.fgt_maxlength]

			print("{:.2%}".format(i * 1.0 / len(examples)))

	def Get_Glove_Representation_Tmp(self, examples):
		glove_vocabulary = {}
		Len = 300

		word_glove_map = {}

		for line in open("./sources/glove_"+str(Len)+"d.dat").read().split("\n"):
			eles = line.split(" ")
			if(len(eles)==Len+1):
				word = eles[0]
				vector = [float(ele) for ele in eles[1:]]
				glove_vocabulary[word] = vector

		for sentence in examples:
			for word in sentence:
				if (word in glove_vocabulary.keys()):
					word_glove_map[word] = glove_vocabulary[word]
				else:
					word_glove_map[word] = glove_vocabulary[self.default_word]

		return word_glove_map
