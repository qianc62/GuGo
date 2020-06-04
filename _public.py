# import matplotlib.pyplot as plt
# import math
import numpy as np
# import time as time
import sklearn.metrics as metrics
import pickle
import nltk
from scipy import stats


# datasets = ["allrecipes", "ifixit", "subwebis", "ProPara", "thsa"]
datasets = ["allrecipes"]
# datasets = ["ifixit"]
dataset = ""
language_channel_num = 6
fgt_maxlength = 30
word_maxlength = 10
label_histogram_x, label_histogram_y = [], []

vocabulary = set()
punctuations = ['\t', ' ', '"', '#', "'", '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '|', '·', '“', '”', '℃', '、', '。', '（', '）', '，', '：', '；']
INF = 999999999.9

seeds_channels = []
example = None

noise_ratio = 1.0

metric_name = ""
metric_value = 0.0

tag_tmp = ["AC", "IN", "NO"]

# # CRF有关全局变量
# global_weight = 0.80
# local_weight = 1.00 - global_weight
# positive_weight = +1.0
# zero_weight = 0.0
# negative_weight = -1.0

# qc_bert_input = None
qc_bert_tokenized_text = []

# with open("./wackypedia_bert_vecs", 'rb') as file:
# 	wackypedia_bert_vecs = pickle.load(file)
# print(len(wackypedia_bert_vecs.keys()), "words")

# with open("./bestmodel", 'rb') as file:
# 	[v2i, mus, lvs] = pickle.load(file)
# print(len(v2i.keys()), "words (bestmodel)")

# with open("./finalmodel", 'rb') as file:
# 	[mus, lvs] = pickle.load(file)
# print(len(mus), "words (finalmodel)")

# with open("./v2i_bert_1024_map", 'rb') as file:
# 	bert1024 = pickle.load(file)
# print(len(bert1024.keys()), "words (bert)")

hidden_states = None

# 是否是一个纯英文字母组成的"word"
def Is_Word(word):
	for ch in word:
		if(not ((ch>='a' and ch<='z') or (ch>='A' and ch<='Z'))):
			return False
	return True

# 是否是一个纯数字组成的"number"
def Is_Number(num):
	for ch in num:
		if (not (ch>='0' and ch<='9')):
			return False
	return True

def Is_LabelText(text):
	for ch in text:
		if (Is_Word(ch)):
			return True
	return False

# # 输出一行虚线
def Print_Dotted_Line(title=""):
	print("--------------------------------------------------"+title+"--------------------------------------------------")

#
# # 是否是一样的一维向量（长度、元素）
# def Same_Array(a1, a2):
# 	if(len(a1)!=len(a2)):
# 		return False
# 	for i in range(len(a1)):
# 		if(a1[i]!=a2[i]):
# 			return False
# 	return True
#
def Max_Index(array):
	max_index = 0
	for i in range(len(array)):
		if(array[i]>array[max_index]):
			max_index = i
	return max_index

def Get_Dotted_Line(title=""):
	return "--------------------------------------------------"+title+"--------------------------------------------------"

def Get_Tabs(k):
	string = ""
	for i in range(k):
		string += "\t"
	return string

# 输出二维矩阵的元素之和
def Matrix_Sum(mat):
	sum = 0.0
	for line in mat:
		for ele in line:
			sum += ele
	return sum
#
# # softmax归一化操作
# def Softmax(array):
# 	array = np.array(array)
# 	array = np.exp(array)
# 	array /= np.sum(array)
# 	return array
#
# # 句子是否"以.和;结束"
# def Sentence_Ends(str):
# 	if(("." in str) or (";" in str)):
# 		return True
# 	return False

def Print_Matrix(mat):
	Print_Dotted_Line()
	for line in mat:
		for ele in line:
			print("%3d  " % ele, end="")
		print()
#
def Map_To_Sorted_List(map):
	x, y = [], []
	for item in sorted(map.items(), key=lambda item: item[1], reverse=True):
		x.append(item[0])
		y.append(item[1])
	return x, y

def List_To_Bool_Map( array ):
	map = {}
	for obj in set(array):
		map[obj] = True
	return map

def Get_Report(true_labels, pred_labels):
	recall = metrics.recall_score(true_labels, pred_labels, average='macro')
	precision = metrics.precision_score(true_labels, pred_labels, average='macro')
	macrof1 = metrics.f1_score(true_labels, pred_labels, average='macro')
	microf1 = metrics.f1_score(true_labels, pred_labels, average='micro')
	acc = metrics.accuracy_score(true_labels, pred_labels)

	# auc = metrics.roc_auc_score(true_labels, pred_labels, average='micro')
	# print(auc)

	# mat = metrics.confusion_matrix(true_labels, pred_labels, labels=labels)
	# classification_report = metrics.classification_report(true_labels, pred_labels, labels=labels, digits=digits)
	# return "recall\tprecision\tmacrof1\tmicrof1\tacc\n{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}".format(recall, precision, macrof1, microf1, acc)
	# return "{}\n{}\n{}\n{}\n{}\n".format(line, str(mat), line, str(classification_report), line)

	return recall, precision, macrof1, microf1, acc
	# return "{:.4}\t{:.4}".format(macrof1, microf1)

def Pickle(variable, path):
	with open(path, 'wb') as file:
		pickle.dump(variable, file)

def Choice(array, num):
	ans = []
	for index in np.random.choice(len(array), num, replace=False):
		ans.append(array[index])
	return ans

def Get_POS(sentence):
	tags = []
	text = nltk.word_tokenize(sentence)
	for tag in nltk.pos_tag(text):
		tags.append(tag[1])
	return  tags

def Get_WordPiece(word, collections):
	while(len(word)>0):
		if(word in collections):
			return word
		word = word[:-1]
	return "the"

def Get_Spearman_Correlation(array1, array2):
	(correlation, pvalue) = stats.spearmanr(array1, array2)
	return correlation

def Pickle_Save(variable, path):
	with open(path, 'wb') as file:
		pickle.dump(variable, file)
	print("Pickle Saved {}".format(path))

def Pickle_Read(filepath):
	with open(filepath, 'rb') as file:
		obj = pickle.load(file)
	print("Pickle Read")
	return obj
