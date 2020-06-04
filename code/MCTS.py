import _public as pb
import copy
import random
import feature
import numpy as np



class MCTSNode:
	def __init__(self):
		self.boardl = []
		self.stonel = []
		self.N, self.W, self.Q, self.P = 0, 0.0, 0.0, 0.0
		self.parent = None
		self.children = []

class MCTS:
	def __init__(self, fragments):
		self.fragments = fragments
		self.nodes = set()

		self.cuct = 0.0001
		self.L = 2
		self.state_feature_weight = 0.5
		self.feature_weight = 0.8

		node = MCTSNode()
		node.P = 1.0
		node.boardl = self.fragments
		node.stonel = ["" for _ in self.fragments]
		self.AddElements( None, node )

	def AddElements(self, u, v):
		if(u!=None):
			self.nodes.add(u)
		if(v!=None):
			self.nodes.add(v)
		if(u!=None and v!=None):
			u.children.append(v)
			v.parent = u

	def Select(self):
		# nodes = []
		# leaves = [node for node in self.nodes if len(node.children)==0]
		# maxValue = -pb.INF
		# value_map = {}
		# for node in leaves:
		# 	value = node.Q + self.cuct * node.P * len(node.children) / (1 + node.N)
		# 	value_map[node] = value
		# 	if(value>maxValue):
		# 		maxValue = value
		# for node in leaves:
		# 	if(value_map[node]==maxValue):
		# 		nodes.append(node)
		# return nodes

		leafValues = [node.Q for node in self.nodes if len(node.children)==0]
		maxValue = np.max(leafValues)
		nodes = [node for node in self.nodes if node.Q==maxValue]
		return nodes

	def Expand(self, nodes):
		queue = []
		queue.extend(nodes)

		for _ in range(self.L):
			children = []
			for node in queue:
				empty_coordinates = [coordinate for coordinate, stone in enumerate(node.stonel) if stone==""]

				# if(len(empty_coordinates)>0):
				# 	empty_coordinates = [empty_coordinates[0]]

				# if(len(empty_coordinates)>0):
				# 	empty_coordinates = [empty_coordinates[len(empty_coordinates)-1]]

				# if(len(empty_coordinates)>2):
				# 	empty_coordinates = [empty_coordinates[0],empty_coordinates[len(empty_coordinates)-1]]

				for coordinate in empty_coordinates:
					for type in pb.label_histogram_x:
						child = MCTSNode()
						child.boardl = node.boardl
						child.stonel  = copy.deepcopy(node.stonel)
						child.stonel[coordinate] = type
						child.P = 1.0 / (len(empty_coordinates)*len(pb.label_histogram_x))
						self.AddElements(node, child)
						children.append(child)
			queue = children

	def Evaluate(self):
		leaves = [node for node in self.nodes if len(node.children)==0]
		for node in leaves:
			node.N += 1
			state_feature, trans_feature = feature.Feature_Triggered_Values([obj.fgt_channels[0] for obj in node.boardl], node.stonel)
			node.W = self.state_feature_weight * state_feature + (1.0-self.state_feature_weight)*trans_feature
			node.Q = node.W / node.N

			# print(node.stonel, node.W)

	def Backup(self):
		leaves = [node for node in self.nodes if len(node.children)==0]
		for node in leaves:
			while(node!=None and node.parent!=None):
				node.parent.N += 1
				node.parent.W += node.W
				node = node.parent

		for node in self.nodes:
			node.Q = node.W / node.N

	def Play(self):
		root = [node for node in self.nodes if node.parent==None][0]

		Qs, Ps = [], []
		for node in root.children:
			Qs.append(node.Q)
			sum = 0.0
			for i in range(len(node.boardl)):
				if(node.stonel[i]!=""):
					sum += node.boardl[i].mssm_probdis[pb.label_histogram_x.index(node.stonel[i])]
			Ps.append(sum)

			# print(node.stonel, node.Q, sum)

		Qs = np.exp(Qs) / np.sum(np.exp(Qs))
		Ps = np.exp(Ps) / np.sum(np.exp(Ps))
		As = self.feature_weight * Qs + (1.0-self.feature_weight) * Ps

		if(len(root.children)==0):
			pb.Print_Dotted_Line("Error")

		bestIndex = 0
		for i in range(len(As)):
			if (As[i]>As[bestIndex]):
				bestIndex = i
		bestNode = root.children[bestIndex]

		keep_nodes = [bestNode]
		keep_nodes.extend(bestNode.children)

		bestNode.parent = None
		# bestNode.N, bestNode.W, bestNode.Q, bestNode.P = 0, 0.0, 0.0, 1.0
		for node in bestNode.children:
			# bestNode.N, bestNode.W, bestNode.Q, bestNode.P = 0, 0.0, 0.0, 1.0/len(bestNode.children)
			node.children.clear()

		self.nodes = set(keep_nodes)

		return bestNode

	def MCTS_Predict(self):
		state = None
		for _ in self.fragments:
			# self.Print([node for node in self.nodes if node.parent==None][0], 0)
			# pb.Print_Dotted_Line()

			nodes = self.Select()
			self.Expand(nodes)
			# self.Print([node for node in self.nodes if node.parent == None][0], 0)
			# pb.Print_Dotted_Line()

			self.Evaluate()

			# pb.Print_Dotted_Line()

			self.Backup()
			state = self.Play()
			# print(state.stonel)
			# self.Print([node for node in self.nodes if node.parent == None][0], 0)
			# pb.Print_Dotted_Line()

		for i in range(len(self.fragments)):
			self.fragments[i].mcts_label = state.stonel[i]
			# if(self.fragments[i].mssm_label != self.fragments[i].mcts_label):
			# 	self.fragments[i].Print()


	def Print(self, node, depth):
		print("{}{}".format(pb.Get_Tabs(depth), node.stonel))
		for child in node.children:
			self.Print(child, depth+1)
