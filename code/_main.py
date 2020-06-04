import _public as pb
import os
import sys
import dataset
import representation
import model
import MCTS
import torch
import time
import numpy as np
import pickle
import math



# import crf
# import plot as fig
# import re
# import model
# import feature as ft
# import crf
# import threading
# import time
# import model_baseline
# import sklearn.metrics as metrics



def main(argv):
	examples_train, examples_test = dataset.Read_Data("./data/"+pb.dataset+".csv")
	print(len(examples_train), len(examples_test))
	dataset.Print_Examples(examples_train)
	dataset.Print_Examples(examples_test)

	dataset.Init_All_Variables(examples_train)

	textEmd = representation.TextEmbedding()
	textEmd.Get_Bert_Representation(examples_train, examples_test)
	pb.Pickle([example.bert_mat for example in examples_train], "./data/"+pb.dataset+"_train_bert")
	pb.Pickle([example.bert_mat for example in examples_test], "./data/"+pb.dataset+"_test_bert")

	examples_train = dataset.Get_Balanced_Data(examples_train)

	xs_train, v1s_train, v2s_train, v3s_train, v4s_train, v5s_train, v6s_train, ys_train = dataset.Get_Encoded_Data(examples_train)
	xs_test,  v1s_test,  v2s_test,  v3s_test,  v4s_test,  v5s_test,  v6s_test,  ys_test  = dataset.Get_Encoded_Data(examples_test)

	documents_xs_train, documents_ys_train, documents_train = [], [], []
	begin_index = 0
	while (begin_index < len(examples_train)):
		end_index = begin_index
		while (end_index < len(examples_train) and examples_train[end_index].group==examples_train[begin_index].group):
			end_index += 1
		documents_xs_train.append(xs_train[begin_index:end_index])
		documents_ys_train.append(ys_train[begin_index:end_index])

		documents_train.append(examples_train[begin_index:end_index])
		begin_index = end_index

	documents_xs_test, documents_ys_test, documents_test = [], [], []
	begin_index = 0
	while (begin_index < len(examples_test)):
		end_index = begin_index
		while (end_index < len(examples_test) and examples_test[end_index].group == examples_test[
			begin_index].group):
			end_index += 1
		documents_xs_test.append(xs_test[begin_index:end_index])
		documents_ys_test.append(ys_test[begin_index:end_index])

		documents_test.append(examples_test[begin_index:end_index])
		begin_index = end_index

	for Times in range(2):
		gual = model.GUAL(xs_train, v1s_train, v2s_train, v3s_train, v4s_train, v5s_train, v6s_train, ys_train)

		try:
			gual.train(documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test)
		finally:
			gual.test(documents_test, documents_xs_test)

			begin_index = 0
			while (begin_index < len(examples_test)):
				end_index = begin_index
				while (end_index < len(examples_test) and examples_test[end_index].group == examples_test[begin_index].group):
					end_index += 1
				fragments = examples_test[begin_index:end_index]

				mcts = MCTS.MCTS(fragments)
				mcts.MCTS_Predict()

				begin_index = end_index

			pb.Print_Dotted_Line()
			# recall, precision, macrof1, microf1, acc = pb.Get_Report([e.label for e in examples_test], [e.mssm_label for e in examples_test], pb.label_histogram_x, 2)
			# print("{:.4f}\t{:.4f}".format(macrof1, microf1))

			ts = [e.label for e in examples_test]
			ps = [e.mcts_label for e in examples_test]
			# recall, precision, macrof1, microf1, acc = pb.Get_Report(ts, ps, pb.label_histogram_x, 2)
			recall, precision, macrof1, microf1, acc = pb.Get_Report(ts, ps)
			print("{:.4f}\t{:.4f}".format(macrof1, microf1))
			pb.Print_Dotted_Line()

if __name__ == "__main__":
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
	# matplotlib.use('TkAgg')
	# print(math.log10(0.0))

	for _dataset in pb.datasets:
		pb.dataset = _dataset
		main(sys.argv)
