import _public as pb
import numpy as np
import torch
from torch import optim
import plot as fig
import dataset
import time



class MC(torch.nn.Module):
	def __init__(self, xs, ys):
		super(MC, self).__init__()

		self.epochs = 301
		self.batch_size = 64
		self.print_delta = 50
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.size)
		# last_dimension = x.size(len(x.size)-1)
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
					print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class RWMD_CC(torch.nn.Module):

	def distance(self, mat1, mat2):
		mat1 = np.array(mat1)
		shape1 = mat1.shape
		v1 = mat1.reshape((-1, ))

		mat2 = np.array(mat2)
		shape2 = mat2.shape
		v2 = mat2.reshape((-1, ))

		dis = np.sum( np.abs( v1 - v2 ) )

		return dis

	def test(self, examples_train, examples_test):
		for e1 in examples_test:
			minDis, bestExample = pb.INF, None
			for e2 in examples_train:
				dis = self.distance(e1.word2vec_mat, e2.word2vec_mat)
				if(dis < minDis):
					minDis = dis
					bestExample = e2
			e1.mssm_label = bestExample.label

		true_labels = [e.label for e in examples_test]
		pred_labels = [e.mssm_label for e in examples_test]
		recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pred_labels, pb.label_histogram_x, 2)
		print("{:.4f}\t{:.4f}".format(macrof1, microf1))

class TextCharCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextCharCNN, self).__init__()

		self.epochs = 5001
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []
		self.in_channels = 1
		self.out_channels = 2
		self.windows = [2, 3, 4]
		self.height = np.array(xs).shape[2]
		self.width  = np.array(xs).shape[3]

		self.convs = torch.nn.ModuleList([
				torch.nn.Sequential( torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(h, self.width), stride=(1, self.width), padding=0),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=(self.height-h+1, 1), stride=(self.height-h+1, 1))
			) for h in [2, 3, 4] ])

		self.fc = torch.nn.Linear( in_features=len(self.windows)*self.out_channels, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = torch.cat([conv(x) for conv in self.convs], dim=1)
		x = x.view(-1, x.size(1))
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

class TextWordCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextWordCNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []
		self.in_channels = 1
		self.out_channels = 2
		self.windows = [2, 3, 4]
		self.height = np.array(xs).shape[2]
		self.width  = np.array(xs).shape[3]
		self.rep_width = self.height * self.width

		self.convs = torch.nn.ModuleList([
				torch.nn.Sequential( torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(h, self.width), stride=(1, self.width), padding=0),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=(self.height-h+1, 1), stride=(self.height-h+1, 1))
			) for h in [2, 3, 4] ])

		self.fc = torch.nn.Linear( in_features=len(self.windows)*self.out_channels, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = torch.cat([conv(x) for conv in self.convs], dim=1)
		x = x.view(-1, x.size(1))
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

	def Print(self):
		for pa in self.parameters():
			print(pa)

class TextRNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextRNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 32
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 64
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []

		self.rnn = torch.nn.RNN( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True)
		self.fc = torch.nn.Linear(self.hidden_size, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = self.fc(out)
		out = out[:, -1, :]
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		# for i,x in enumerate(xs_test):
		# 	print(i, np.array(x))
		# x = np.array(xs_test)

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		# optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TextBiLSTM(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextBiLSTM, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 200
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []
		self.rnn = torch.nn.LSTM( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)
		self.fc = torch.nn.Linear(self.hidden_size*2, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = self.fc(out)
		out = out[:, -1, :]
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TextRCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextRCNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 200
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []

		self.rnn = torch.nn.LSTM(input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.height = np.array(xs).shape[1]
		self.width = self.hidden_size * 2
		self.rep_width = self.height*2

		self.convs = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, self.width), stride=(1, self.width), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
		)

		self.fc = torch.nn.Linear(self.rep_width, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = out.unsqueeze(1)
		out = self.convs(out)
		out = out.view(-1, out.size(1)*out.size(2)*out.size(3))
		out = self.fc(out)
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class Word2Vec(torch.nn.Module):
	def __init__(self, xs, ys):
		super(Word2Vec, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class Glove(torch.nn.Module):
	def __init__(self, xs, ys):
		super(Glove, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class FastText(torch.nn.Module):
	def __init__(self, xs, ys):
		super(FastText, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.size)
		# last_dimension = x.size(len(x.size)-1)
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class ELMo(torch.nn.Module):
	def __init__(self, xs, ys):
		super(ELMo, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TransformerXL(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TransformerXL, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class GPT2(torch.nn.Module):
	def __init__(self, xs, ys):
		super(GPT2, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class BERT(torch.nn.Module):
	def __init__(self, xs, ys):
		super(BERT, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 1
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				# print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

				if(pb.metric_name=="ma" and max_macrof1>=pb.metric_value):
					return max_macrof1, max_microf1
				if (pb.metric_name=="mi" and max_microf1>=pb.metric_value):
					return max_macrof1, max_microf1

		return max_macrof1, max_microf1

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class MSSM_CPCM(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(MSSM_CPCM, self).__init__()

		self.epochs = 5001
		self.batch_size = 32
		self.print_delta = 1000
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		# print(np.array(xs).shape)
		self.rep_width = np.array(xs).shape[1] * np.array(xs).shape[2] * np.array(xs).shape[3]
		self.mssm_width = len(v1s[0])+len(v2s[0])+len(v3s[0])+len(v4s[0])+len(v5s[0])+len(v6s[0])

		self.attention1 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v1s[0]), out_features=len(v1s[0])),
			torch.nn.Sigmoid()
		)
		self.attention2 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v2s[0]), out_features=len(v2s[0])),
			torch.nn.Sigmoid()
		)
		self.attention3 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v3s[0]), out_features=len(v3s[0])),
			torch.nn.Sigmoid()
		)
		self.attention4 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v4s[0]), out_features=len(v4s[0])),
			torch.nn.Sigmoid()
		)
		self.attention5 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v5s[0]), out_features=len(v5s[0])),
			torch.nn.Sigmoid()
		)
		self.attention6 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v6s[0]), out_features=len(v6s[0])),
			torch.nn.Sigmoid()
		)

		self.fc = torch.nn.Linear( in_features=self.rep_width+self.mssm_width, out_features=len(pb.label_histogram_x) )
		# self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x, v1, v2, v3, v4, v5, v6):
		x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
		v = x

		v1 = v1.view(-1, v1.size(1))
		v2 = v2.view(-1, v2.size(1))
		v3 = v3.view(-1, v3.size(1))
		v4 = v4.view(-1, v4.size(1))
		v5 = v5.view(-1, v5.size(1))
		v6 = v6.view(-1, v6.size(1))

		g1 = self.attention1(v1)
		g2 = self.attention2(v2)
		g3 = self.attention3(v3)
		g4 = self.attention4(v4)
		g5 = self.attention5(v5)
		g6 = self.attention6(v6)

		v1 = g1.mul(v1)
		v2 = g2.mul(v2)
		v3 = g3.mul(v3)
		v4 = g4.mul(v4)
		v5 = g5.mul(v5)
		v6 = g6.mul(v6)

		v = torch.cat((v, v1), dim=1)
		v = torch.cat((v, v2), dim=1)
		v = torch.cat((v, v3), dim=1)
		v = torch.cat((v, v4), dim=1)
		v = torch.cat((v, v5), dim=1)
		v = torch.cat((v, v6), dim=1)

		o = self.fc(v)

		return o

	def train(self, xs_train, v1s_train, v2s_train, v3s_train, v4s_train, v5s_train, v6s_train, ys_train, xs_test, v1s_test, v2s_test, v3s_test, v4s_test, v5s_test, v6s_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
		all_v1s_test = torch.autograd.Variable(torch.Tensor(np.array(v1s_test)))
		all_v2s_test = torch.autograd.Variable(torch.Tensor(np.array(v2s_test)))
		all_v3s_test = torch.autograd.Variable(torch.Tensor(np.array(v3s_test)))
		all_v4s_test = torch.autograd.Variable(torch.Tensor(np.array(v4s_test)))
		all_v5s_test = torch.autograd.Variable(torch.Tensor(np.array(v5s_test)))
		all_v6s_test = torch.autograd.Variable(torch.Tensor(np.array(v6s_test)))

		# try:
		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
		# optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_v1s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v1s_train) if i in rand_index])))
			batch_v2s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v2s_train) if i in rand_index])))
			batch_v3s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v3s_train) if i in rand_index])))
			batch_v4s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v4s_train) if i in rand_index])))
			batch_v5s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v5s_train) if i in rand_index])))
			batch_v6s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v6s_train) if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs, batch_v1s, batch_v2s, batch_v3s, batch_v4s, batch_v5s, batch_v6s)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test, all_v1s_test, all_v2s_test, all_v3s_test, all_v4s_test, all_v5s_test, all_v6s_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
		# finally:
		# 	string = ""
		# 	string += "Dataset: {}\n".format(pb.dataset)
		# 	string += "language_channel_num: {}\n".format(pb.language_channel_num)
		# 	string += "Class Name: {}\n".format(self.__class__.__name__)
		# 	string += "epochs: {}\n".format(self.epochs)
		# 	string += "batch_size: {}\n".format(self.batch_size)
		# 	string += "print_delta: {}\n".format(self.print_delta)
		# 	string += "learning_rate: {:.4f}\n".format(self.learning_rate)
		# 	string += "hidden_nn_num: {}\n".format(self.hidden_nn_num)
		# 	string += "dropout_ratio: {:.2f}\n".format(self.dropout_ratio)
		# 	# string += "in_channels: {}\n".format(self.in_channels)
		# 	# string += "out_channels: {}\n".format(self.out_channels)
		# 	# string += "windows: {}\n".format(self.windows)
		# 	string += "rep_width: {}\n".format(self.rep_width)
		# 	string += "mssm_width: {}\n".format(self.mssm_width)
		# 	fig.Plot_Two_Lines(self.train_acc_list, self.test_acc_list, string)

		return max_macrof1, max_microf1

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			all_v1s = torch.autograd.Variable(torch.Tensor(np.array(v1s)))
			all_v2s = torch.autograd.Variable(torch.Tensor(np.array(v2s)))
			all_v3s = torch.autograd.Variable(torch.Tensor(np.array(v3s)))
			all_v4s = torch.autograd.Variable(torch.Tensor(np.array(v4s)))
			all_v5s = torch.autograd.Variable(torch.Tensor(np.array(v5s)))
			all_v6s = torch.autograd.Variable(torch.Tensor(np.array(v6s)))
			example.mssm_probdis = self.forward(all_xs, all_v1s, all_v2s, all_v3s, all_v4s, all_v5s, all_v6s).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

	def Print(self):
		for pa in self.parameters():
			print(pa)

class GUAL(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(GUAL, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size*2, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		print(x.shape)

		v, _ = self.bilstm(x, None)
		# print(v.shape)

		v = v.view(v.shape[1], -1)
		# print(v.shape)

		o = self.mlp(v)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class RNN(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(RNN, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.rnn = torch.nn.RNN(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.shape)

		o, _ = self.rnn(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTM(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(LSTM, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.lstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.shape)

		o, _ = self.lstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTMCRF(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(LSTMCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

		self.emission   = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size,self.bilstm_hidden_size), requires_grad=True)
		self.transition = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size,self.bilstm_hidden_size), requires_grad=True)

	def crf(self, o):
		O = None

		for i in range(o.shape[0]):
			if(i==0):
				u = torch.autograd.Variable(torch.Tensor(np.zeros(self.bilstm_hidden_size)))
			else:
				u = o[i-1]

			v = o[i]
			# print(v.shape)

			u = torch.unsqueeze(u, 0)
			v = torch.unsqueeze(v, 0)

			w1 = v.mm(self.emission)
			w2 = u.mm(self.transition).add(v)
			w = w1.add(w2)
			# print(w1.shape)
			# print(w2.shape)
			# print(w.shape)

			if (i == 0):
				O = w
			else:
				O = torch.cat((O, w), 0)
				# print(O.shape)

		return O

	def forward(self, x):
		# print(x.shape)

		o, _ = self.bilstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.crf(o)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class BILSTMCRF(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(BILSTMCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size*2, out_features=len(pb.label_histogram_x) )

		self.emission   = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size*2,self.bilstm_hidden_size*2), requires_grad=True)
		self.transition = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size*2,self.bilstm_hidden_size*2), requires_grad=True)

	def crf(self, o):
		O = None

		for i in range(o.shape[0]):
			if(i==0):
				u = torch.autograd.Variable(torch.Tensor(np.zeros(self.bilstm_hidden_size*2)))
			else:
				u = o[i-1]

			v = o[i]
			# print(v.shape)

			u = torch.unsqueeze(u, 0)
			v = torch.unsqueeze(v, 0)

			w1 = v.mm(self.emission)
			w2 = u.mm(self.transition).add(v)
			w = w1.add(w2)
			# print(w1.shape)
			# print(w2.shape)
			# print(w.shape)

			if (i == 0):
				O = w
			else:
				O = torch.cat((O, w), 0)
				# print(O.shape)

		return O

	def forward(self, x):
		# print(x.shape)

		o, _ = self.bilstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.crf(o)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTMCNNCRF(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(LSTMCNNCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		)

		self.lstm = torch.nn.LSTM(input_size=int(self.rep_width/4), hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.lstm(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class CNNDAC(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(CNNDAC, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		)

		self.rnn = torch.nn.RNN(input_size=int(self.rep_width/4), hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.rnn(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TLRCN(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(TLRCN, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 2

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, self.rep_width), stride=(1, self.rep_width), padding=0),
			torch.nn.ReLU(),
		)

		self.rnn = torch.nn.RNN(input_size=1, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.rnn(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class PARPOS(torch.nn.Module):
	def __init__(self, xs, ys):
		super(PARPOS, self).__init__()

		self.epochs = 5001
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):
		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

import _public as pb
import numpy as np
import torch
from torch import optim
import plot as fig
import dataset



class MC(torch.nn.Module):
	def __init__(self, xs, ys):
		super(MC, self).__init__()

		self.epochs = 301
		self.batch_size = 64
		self.print_delta = 50
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.size)
		# last_dimension = x.size(len(x.size)-1)
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
					print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class RWMD_CC(torch.nn.Module):

	def distance(self, mat1, mat2):
		mat1 = np.array(mat1)
		shape1 = mat1.shape
		v1 = mat1.reshape((-1, ))

		mat2 = np.array(mat2)
		shape2 = mat2.shape
		v2 = mat2.reshape((-1, ))

		dis = np.sum( np.abs( v1 - v2 ) )

		return dis

	def test(self, examples_train, examples_test):
		for e1 in examples_test:
			minDis, bestExample = pb.INF, None
			for e2 in examples_train:
				dis = self.distance(e1.word2vec_mat, e2.word2vec_mat)
				if(dis < minDis):
					minDis = dis
					bestExample = e2
			e1.mssm_label = bestExample.label

		true_labels = [e.label for e in examples_test]
		pred_labels = [e.mssm_label for e in examples_test]
		recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pred_labels, pb.label_histogram_x, 2)
		print("{:.4f}\t{:.4f}".format(macrof1, microf1))


class TextCharCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextCharCNN, self).__init__()

		self.epochs = 5001
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []
		self.in_channels = 1
		self.out_channels = 2
		self.windows = [2, 3, 4]
		self.height = np.array(xs).shape[2]
		self.width  = np.array(xs).shape[3]

		self.convs = torch.nn.ModuleList([
				torch.nn.Sequential( torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(h, self.width), stride=(1, self.width), padding=0),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=(self.height-h+1, 1), stride=(self.height-h+1, 1))
			) for h in [2, 3, 4] ])

		self.fc = torch.nn.Linear( in_features=len(self.windows)*self.out_channels, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = torch.cat([conv(x) for conv in self.convs], dim=1)
		x = x.view(-1, x.size(1))
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

class TextWordCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextWordCNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []
		self.in_channels = 1
		self.out_channels = 2
		self.windows = [2, 3, 4]
		self.height = np.array(xs).shape[2]
		self.width  = np.array(xs).shape[3]
		self.rep_width = self.height * self.width

		self.convs = torch.nn.ModuleList([
				torch.nn.Sequential( torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(h, self.width), stride=(1, self.width), padding=0),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=(self.height-h+1, 1), stride=(self.height-h+1, 1))
			) for h in [2, 3, 4] ])

		self.fc = torch.nn.Linear( in_features=len(self.windows)*self.out_channels, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = torch.cat([conv(x) for conv in self.convs], dim=1)
		x = x.view(-1, x.size(1))
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

	def Print(self):
		for pa in self.parameters():
			print(pa)

class TextRNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextRNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 32
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 64
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []

		self.rnn = torch.nn.RNN( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True)
		self.fc = torch.nn.Linear(self.hidden_size, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = self.fc(out)
		out = out[:, -1, :]
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		# for i,x in enumerate(xs_test):
		# 	print(i, np.array(x))
		# x = np.array(xs_test)

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		# optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TextBiLSTM(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextBiLSTM, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 200
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []
		self.rnn = torch.nn.LSTM( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)
		self.fc = torch.nn.Linear(self.hidden_size*2, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = self.fc(out)
		out = out[:, -1, :]
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TextRCNN(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TextRCNN, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.w2v_size = 100
		self.hidden_size = 200
		self.layer_num = 1
		self.train_acc_list = []
		self.test_acc_list = []

		self.rnn = torch.nn.LSTM(input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.height = np.array(xs).shape[1]
		self.width = self.hidden_size * 2
		self.rep_width = self.height*2

		self.convs = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, self.width), stride=(1, self.width), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
		)

		self.fc = torch.nn.Linear(self.rep_width, len(pb.label_histogram_x))

	def forward(self, x):
		out, _ = self.rnn(x, None)
		out = out.unsqueeze(1)
		out = self.convs(out)
		out = out.view(-1, out.size(1)*out.size(2)*out.size(3))
		out = self.fc(out)
		return out

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class Word2Vec(torch.nn.Module):
	def __init__(self, xs, ys):
		super(Word2Vec, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class Glove(torch.nn.Module):
	def __init__(self, xs, ys):
		super(Glove, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class FastText(torch.nn.Module):
	def __init__(self, xs, ys):
		super(FastText, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.size)
		# last_dimension = x.size(len(x.size)-1)
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class ELMo(torch.nn.Module):
	def __init__(self, xs, ys):
		super(ELMo, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TransformerXL(torch.nn.Module):
	def __init__(self, xs, ys):
		super(TransformerXL, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class GPT2(torch.nn.Module):
	def __init__(self, xs, ys):
		super(GPT2, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class BERT(torch.nn.Module):
	def __init__(self, xs, ys):
		super(BERT, self).__init__()

		self.epochs = 4501
		self.batch_size = 64
		self.print_delta = 50
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

				# if(pb.metric_name=="ma" and max_macrof1>=pb.metric_value):
				# 	return max_macrof1, max_microf1
				# if (pb.metric_name=="mi" and max_microf1>=pb.metric_value):
				# 	return max_macrof1, max_microf1

		return max_macrof1, max_microf1

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class MSSM_CPCM(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(MSSM_CPCM, self).__init__()

		self.epochs = 1500
		self.batch_size = 32
		self.print_delta = 1
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		# print(np.array(xs).shape)
		self.rep_width = np.array(xs).shape[1] * np.array(xs).shape[2] * np.array(xs).shape[3]
		self.mssm_width = len(v1s[0])+len(v2s[0])+len(v3s[0])+len(v4s[0])+len(v5s[0])+len(v6s[0])

		self.attention1 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v1s[0]), out_features=len(v1s[0])),
			torch.nn.Sigmoid()
		)
		self.attention2 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v2s[0]), out_features=len(v2s[0])),
			torch.nn.Sigmoid()
		)
		self.attention3 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v3s[0]), out_features=len(v3s[0])),
			torch.nn.Sigmoid()
		)
		self.attention4 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v4s[0]), out_features=len(v4s[0])),
			torch.nn.Sigmoid()
		)
		self.attention5 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v5s[0]), out_features=len(v5s[0])),
			torch.nn.Sigmoid()
		)
		self.attention6 = torch.nn.Sequential(
			torch.nn.Linear(in_features=len(v6s[0]), out_features=len(v6s[0])),
			torch.nn.Sigmoid()
		)

		self.fc = torch.nn.Linear( in_features=self.rep_width+self.mssm_width, out_features=len(pb.label_histogram_x) )
		# self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x, v1, v2, v3, v4, v5, v6):
		x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
		v = x

		v1 = v1.view(-1, v1.size(1))
		v2 = v2.view(-1, v2.size(1))
		v3 = v3.view(-1, v3.size(1))
		v4 = v4.view(-1, v4.size(1))
		v5 = v5.view(-1, v5.size(1))
		v6 = v6.view(-1, v6.size(1))

		g1 = self.attention1(v1)
		g2 = self.attention2(v2)
		g3 = self.attention3(v3)
		g4 = self.attention4(v4)
		g5 = self.attention5(v5)
		g6 = self.attention6(v6)

		v1 = g1.mul(v1)
		v2 = g2.mul(v2)
		v3 = g3.mul(v3)
		v4 = g4.mul(v4)
		v5 = g5.mul(v5)
		v6 = g6.mul(v6)

		v = torch.cat((v, v1), dim=1)
		v = torch.cat((v, v2), dim=1)
		v = torch.cat((v, v3), dim=1)
		v = torch.cat((v, v4), dim=1)
		v = torch.cat((v, v5), dim=1)
		v = torch.cat((v, v6), dim=1)

		o = self.fc(v)

		return o

	def train(self, xs_train, v1s_train, v2s_train, v3s_train, v4s_train, v5s_train, v6s_train, ys_train, xs_test, v1s_test, v2s_test, v3s_test, v4s_test, v5s_test, v6s_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
		all_v1s_test = torch.autograd.Variable(torch.Tensor(np.array(v1s_test)))
		all_v2s_test = torch.autograd.Variable(torch.Tensor(np.array(v2s_test)))
		all_v3s_test = torch.autograd.Variable(torch.Tensor(np.array(v3s_test)))
		all_v4s_test = torch.autograd.Variable(torch.Tensor(np.array(v4s_test)))
		all_v5s_test = torch.autograd.Variable(torch.Tensor(np.array(v5s_test)))
		all_v6s_test = torch.autograd.Variable(torch.Tensor(np.array(v6s_test)))

		# try:
		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
		# optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		mcts_start_time = time.time()
		times = np.zeros(101)
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_v1s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v1s_train) if i in rand_index])))
			batch_v2s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v2s_train) if i in rand_index])))
			batch_v3s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v3s_train) if i in rand_index])))
			batch_v4s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v4s_train) if i in rand_index])))
			batch_v5s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v5s_train) if i in rand_index])))
			batch_v6s = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(v6s_train) if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs, batch_v1s, batch_v2s, batch_v3s, batch_v4s, batch_v5s, batch_v6s)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test, all_v1s_test, all_v2s_test, all_v3s_test, all_v4s_test, all_v5s_test, all_v6s_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

				duration = time.time() - mcts_start_time
				index = int(100*max_microf1)
				if(times[index]==0.0):
					times[index] = duration
		# finally:
		# 	string = ""
		# 	string += "Dataset: {}\n".format(pb.dataset)
		# 	string += "language_channel_num: {}\n".format(pb.language_channel_num)
		# 	string += "Class Name: {}\n".format(self.__class__.__name__)
		# 	string += "epochs: {}\n".format(self.epochs)
		# 	string += "batch_size: {}\n".format(self.batch_size)
		# 	string += "print_delta: {}\n".format(self.print_delta)
		# 	string += "learning_rate: {:.4f}\n".format(self.learning_rate)
		# 	string += "hidden_nn_num: {}\n".format(self.hidden_nn_num)
		# 	string += "dropout_ratio: {:.2f}\n".format(self.dropout_ratio)
		# 	# string += "in_channels: {}\n".format(self.in_channels)
		# 	# string += "out_channels: {}\n".format(self.out_channels)
		# 	# string += "windows: {}\n".format(self.windows)
		# 	string += "rep_width: {}\n".format(self.rep_width)
		# 	string += "mssm_width: {}\n".format(self.mssm_width)
		# 	fig.Plot_Two_Lines(self.train_acc_list, self.test_acc_list, string)

		print(times)

		fig.Plot_One_Lines(times)

		return max_macrof1, max_microf1

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			all_v1s = torch.autograd.Variable(torch.Tensor(np.array(v1s)))
			all_v2s = torch.autograd.Variable(torch.Tensor(np.array(v2s)))
			all_v3s = torch.autograd.Variable(torch.Tensor(np.array(v3s)))
			all_v4s = torch.autograd.Variable(torch.Tensor(np.array(v4s)))
			all_v5s = torch.autograd.Variable(torch.Tensor(np.array(v5s)))
			all_v6s = torch.autograd.Variable(torch.Tensor(np.array(v6s)))
			example.mssm_probdis = self.forward(all_xs, all_v1s, all_v2s, all_v3s, all_v4s, all_v5s, all_v6s).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

	def Print(self):
		for pa in self.parameters():
			print(pa)

class GUAL(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(GUAL, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 50
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 2

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size*2, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.shape)

		v, _ = self.bilstm(x, None)
		# print(v.shape)

		v = v.view(v.shape[1], -1)
		# print(v.shape)

		o = self.mlp(v)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class RNN(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(RNN, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.rnn = torch.nn.RNN(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.shape)

		o, _ = self.rnn(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTM(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(LSTM, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.lstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print(x.shape)

		o, _ = self.lstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTMCRF(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(LSTMCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

		self.emission   = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size,self.bilstm_hidden_size), requires_grad=True)
		self.transition = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size,self.bilstm_hidden_size), requires_grad=True)

	def crf(self, o):
		O = None

		for i in range(o.shape[0]):
			if(i==0):
				u = torch.autograd.Variable(torch.Tensor(np.zeros(self.bilstm_hidden_size)))
			else:
				u = o[i-1]

			v = o[i]
			# print(v.shape)

			u = torch.unsqueeze(u, 0)
			v = torch.unsqueeze(v, 0)

			w1 = v.mm(self.emission)
			w2 = u.mm(self.transition).add(v)
			w = w1.add(w2)
			# print(w1.shape)
			# print(w2.shape)
			# print(w.shape)

			if (i == 0):
				O = w
			else:
				O = torch.cat((O, w), 0)
				# print(O.shape)

		return O

	def forward(self, x):
		# print(x.shape)

		o, _ = self.bilstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.crf(o)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class BILSTMCRF(torch.nn.Module):
	def __init__(self, xs, v1s, v2s, v3s, v4s, v5s, v6s, ys_train):
		super(BILSTMCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.bilstm = torch.nn.LSTM(input_size=self.rep_width, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size*2, out_features=len(pb.label_histogram_x) )

		self.emission   = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size*2,self.bilstm_hidden_size*2), requires_grad=True)
		self.transition = torch.autograd.Variable(torch.randn(self.bilstm_hidden_size*2,self.bilstm_hidden_size*2), requires_grad=True)

	def crf(self, o):
		O = None

		for i in range(o.shape[0]):
			if(i==0):
				u = torch.autograd.Variable(torch.Tensor(np.zeros(self.bilstm_hidden_size*2)))
			else:
				u = o[i-1]

			v = o[i]
			# print(v.shape)

			u = torch.unsqueeze(u, 0)
			v = torch.unsqueeze(v, 0)

			w1 = v.mm(self.emission)
			w2 = u.mm(self.transition).add(v)
			w = w1.add(w2)
			# print(w1.shape)
			# print(w2.shape)
			# print(w.shape)

			if (i == 0):
				O = w
			else:
				O = torch.cat((O, w), 0)
				# print(O.shape)

		return O

	def forward(self, x):
		# print(x.shape)

		o, _ = self.bilstm(x, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.crf(o)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class LSTMCNNCRF(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(LSTMCNNCRF, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		)

		self.lstm = torch.nn.LSTM(input_size=int(self.rep_width/4), hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.lstm(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class CNNDAC(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(CNNDAC, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 1

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2), padding=0),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
		)

		self.rnn = torch.nn.RNN(input_size=int(self.rep_width/4), hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.rnn(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class TLRCN(torch.nn.Module):
	def __init__(self, xs, ys_train):
		super(TLRCN, self).__init__()

		self.epochs = 2001
		self.batch_size = 2
		self.print_delta = 200
		self.learning_rate = 0.0001

		# self.neighbor_window = 1
		self.bilstm_hidden_size = 400
		self.layer_num = 2

		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, self.rep_width), stride=(1, self.rep_width), padding=0),
			torch.nn.ReLU(),
		)

		self.rnn = torch.nn.RNN(input_size=1, hidden_size=self.bilstm_hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=False)

		self.mlp = torch.nn.Linear( in_features=self.bilstm_hidden_size, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		# print()
		# print(x.shape)

		o = torch.unsqueeze(x, 1)
		# print(o.shape)

		o = self.cnn(o)
		# print(o.shape)

		o = torch.squeeze(o, 1)
		# print(o.shape)

		o, _ = self.rnn(o, None)
		# print(o.shape)

		o = o.view(o.shape[1], -1)
		# print(o.shape)

		o = self.mlp(o)
		# print(o.shape)

		return o

	def train(self, documents_xs_train, documents_ys_train, documents_xs_test, documents_ys_test):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):

			rand_index = np.random.choice(len(documents_xs_train), size=self.batch_size, replace=False)

			for index in rand_index:
				optimizer.zero_grad()

				batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_train[index]])))
				batch_ys = torch.autograd.Variable(torch.LongTensor(np.array(documents_ys_train[index])))

				train_prediction = self.forward(batch_xs)

				loss = criterion(train_prediction, batch_ys)

				loss.backward()

				optimizer.step()

			if (epoch % self.print_delta == 0):
				pre_labels = []
				for index in range(len(documents_xs_test)):
					batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs_test[index]])))
					prediction_test = self.forward(batch_xs)
					pre_labels.extend([pb.Max_Index(line) for line in prediction_test.data.numpy()])

				true_labels = []
				for document_y in documents_ys_test:
					true_labels.extend(document_y)

				recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):

		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class PARPOS(torch.nn.Module):
	def __init__(self, xs, ys):
		super(PARPOS, self).__init__()

		self.epochs = 5001
		self.batch_size = 64
		self.print_delta = 500
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]

		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )

	def forward(self, x):
		x = x.view(-1, self.rep_width)
		o = self.fc(x)
		return o

	def train(self, xs_train, ys_train, xs_test, ys_test):

		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))

	def test(self, documents, documents_xs):
		for index in range(len(documents)):
			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([documents_xs[index]])))
			prediction_test = self.forward(batch_xs).data.numpy()
			for i in range(len(prediction_test)):
				example = documents[index][i]
				example.mssm_probdis = prediction_test[i]
				example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

class Tmp(torch.nn.Module):
	def __init__(self):
		super(Tmp, self).__init__()

		self.epochs = 5001
		self.batch_size = 32
		self.print_delta = 10
		self.learning_rate = 0.0001
		self.hidden_nn_num = 200
		self.train_acc_list = []
		self.test_acc_list = []

		self.fc = torch.nn.Linear( in_features=600, out_features=len(pb.tag_tmp) )

	def forward(self, x, v):
		x = x.view(-1, 300)
		v = v.view(-1, 300)

		o = torch.cat((x, v), dim=1)

		o = self.fc(o)

		return o

	def train(self, xs_train, vs_train, ys_train, xs_test, vs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
		all_vs_test = torch.autograd.Variable(torch.Tensor(np.array(vs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		max_macrof1, max_microf1 = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_vs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(vs_train) if i in rand_index])))
			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs, batch_vs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				prediction_test = self.forward(all_xs_test, all_vs_test)
				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.tag_tmp, 2)
				if (microf1 > max_microf1):
					max_macrof1 = macrof1
					max_microf1 = microf1
				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
		# finally:
		# 	string = ""
		# 	string += "Dataset: {}\n".format(pb.dataset)
		# 	string += "language_channel_num: {}\n".format(pb.language_channel_num)
		# 	string += "Class Name: {}\n".format(self.__class__.__name__)
		# 	string += "epochs: {}\n".format(self.epochs)
		# 	string += "batch_size: {}\n".format(self.batch_size)
		# 	string += "print_delta: {}\n".format(self.print_delta)
		# 	string += "learning_rate: {:.4f}\n".format(self.learning_rate)
		# 	string += "hidden_nn_num: {}\n".format(self.hidden_nn_num)
		# 	string += "dropout_ratio: {:.2f}\n".format(self.dropout_ratio)
		# 	# string += "in_channels: {}\n".format(self.in_channels)
		# 	# string += "out_channels: {}\n".format(self.out_channels)
		# 	# string += "windows: {}\n".format(self.windows)
		# 	string += "rep_width: {}\n".format(self.rep_width)
		# 	string += "mssm_width: {}\n".format(self.mssm_width)
		# 	fig.Plot_Two_Lines(self.train_acc_list, self.test_acc_list, string)

		return max_macrof1, max_microf1

	def test(self, examples):
		for example in examples:
			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
			all_v1s = torch.autograd.Variable(torch.Tensor(np.array(v1s)))
			all_v2s = torch.autograd.Variable(torch.Tensor(np.array(v2s)))
			all_v3s = torch.autograd.Variable(torch.Tensor(np.array(v3s)))
			all_v4s = torch.autograd.Variable(torch.Tensor(np.array(v4s)))
			all_v5s = torch.autograd.Variable(torch.Tensor(np.array(v5s)))
			all_v6s = torch.autograd.Variable(torch.Tensor(np.array(v6s)))
			example.mssm_probdis = self.forward(all_xs, all_v1s, all_v2s, all_v3s, all_v4s, all_v5s, all_v6s).data.numpy()[0]
			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]

	def Get_Accuracy_of_Distributions(self, preddis, ys):
		up, down = 0, 0
		for i in range(len(ys)):
			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
			down += 1
		return up * 1.0 / down

	def Print(self):
		for pa in self.parameters():
			print(pa)

