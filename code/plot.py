import _public as pb
import matplotlib.pyplot as plt
import numpy as np
import time as time
import math



# 绘制直方图的一条bar
def Plot_Bar(x, y):
	plt.xticks(fontsize=8)
	for i in range(len(x)):
		plt.bar([x[i]], [y[i]])

# 绘制直方图
def Plot_Bars(xs, ys):
	plt.xticks(fontsize=8)
	for i in range(len(xs)):
		plt.subplot(len(xs)*100+10+1+i)
		Plot_Bar(xs[i], ys[i])
	plt.show()

# 绘制单高斯曲线
def Plot_Gauss(x, parray):
	y, mu, sigma, a, h = parray[0], parray[1], parray[2], parray[3], parray[4]

	if(x==None):
		x = []
		for i in range(len(y)):
			x.append(i)

	plt.xticks(fontsize=8)

	for i in range(len(x)):
		plt.bar([x[i]], [y[i]])

	x_ = np.arange(0, len(x), 0.01)
	y_ = a * np.exp(-(x_-mu)**2/(2*sigma**2)) + h
	plt.plot(x_, y_)

# 绘制混合高斯曲线
def Plot_Mixture_Gauss(word, mul, sil, al, hl):
	x_ = np.arange(0, len(pb.char_sorted_x), 0.01)
	y_ = sfm.Mixture_Gauss(x_, mul, sil, al, hl)
	plt.plot(x_, y_)
	plt.text(0, np.max(y_)/8, word+" (w="+str(pb.cell_perception_window_size)+")")
	for i in range(len(pb.char_sorted_x)):
		plt.text(i, 0, pb.char_sorted_x[i], size=8)
	plt.savefig("./figs/Mixture_Gauss_Signal_" + str(time.time()) + ".pdf")
	plt.show()

# 绘制字符Cell
def Plot_Cells(mssm):

	for i in range(len(mssm.char_sorted_x)):
		char = mssm.char_sorted_x[i]
		plt.subplot(10, 5, 1+i)
		plt.xlabel("")
		plt.ylabel("")
		plt.xticks(fontsize=6)
		plt.yticks(fontsize=6)
		plt.text(len(mssm.char_sorted_x) / 2, 0.5, char)

		Plot_Gauss(mssm.char_sorted_x, mssm.cell_map[char])
		print(1)

	plt.savefig("./figs/cells_"+str(time.time())+".pdf")
	plt.show()

# 绘制训练曲线和测试曲线
def Plot_One_Lines(values, string=""):
	x = np.linspace(1, len(values), len(values))

	logfunction = math.log2

	# plt.ylim((0, logfunction(100.0)))

	values = [logfunction(v*1.0+0.000000001) for v in values]

	plt.plot(x, values, c="r", ls="-", marker='o', markersize=4, label="Train Acc")

	plt.xlabel('Accuracy')
	plt.ylabel('Time')

	plt.legend()

	# plt.text((np.max(x)+np.min(x))/4.0, 20, string)

	plt.savefig("./figs/time{}.pdf".format(time.time()))
	plt.show()


# 绘制训练曲线和测试曲线
def Plot_Two_Lines(train_values, test_values, string=""):
	x = np.linspace(1, len(train_values), len(train_values))

	plt.ylim((0, 100))

	train_values = [v*100.0 for v in train_values]
	test_values = [v*100.0 for v in test_values]

	plt.plot(x, train_values, c="r", ls="-", marker='o', markersize=4, label="Train Acc")
	plt.plot(x, test_values,  c="g", ls="-", marker='x', markersize=4, label="Test Acc")

	plt.title("Best_Train_Acc: %.2f%%    Best_Test_Acc: %.2f%%" % (np.max(train_values), np.max(test_values)))
	plt.xlabel('Epoches')
	plt.ylabel('Accuracy')

	plt.legend()

	plt.text((np.max(x)+np.min(x))/4.0, 20, string)

	plt.savefig("./figs/best_acc(s): %.2f%%__%.2f%%.pdf" % (np.max(train_values), np.max(test_values)))
	plt.show()
