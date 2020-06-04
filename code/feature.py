import _public as pb
import re



def Get_State_Feature_Vector(xs, ys, state_function):
	feature_vector = []
	for i in range(len(xs)):
		value = 0.0
		if(ys[i]!=""):
			value = state_function(xs[i], ys[i])
		feature_vector.append(value)
	return feature_vector

def Get_Second_Order_Trans_Feature_Vector(xs, ys, trans_function):
	feature_vector = []
	for i in range(len(xs)):
		if(i+1>=len(xs)):
			continue
		value = 0.0
		if(ys[i]!="" and ys[i+1]!=""):
			value = trans_function(xs[i], ys[i], xs[i+1], ys[i+1])
		feature_vector.append(value)
	return feature_vector

def Get_Third_Order_Trans_Feature_Vector(xs, ys, trans_function):
	feature_vector = []
	for i in range(len(xs)):
		if(i+2>=len(xs)):
			continue
		value = 0.0
		if(ys[i]!="" and ys[i+1]!=""):
			value = trans_function(xs[i], ys[i], xs[i+1], ys[i+1], xs[i+2], ys[i+2])
		feature_vector.append(value)
	return feature_vector

def AllRecipes_State_Feature_1(x, y):
	if (x.startswith("after ") or
		x.startswith("if ") or
		x.startswith("during ") or
		x.startswith("until ") or
		x.startswith("meanwhile ") or
		x.startswith("when ") or
		x.startswith("while ") or
		x.startswith("with ") or
		x.startswith("repeat ") or
		x.startswith("once ") ):
		if(y in ["N-"]):
			return 1.0
		else:
			return -1.0
	return 0.0

def AllRecipes_State_Feature_2(x, y):
	if ("should " in x or
		"etc " in x or
		"do not " in x or
		"don ' t " in x or
		"will " in x or
		"keep in mind " in x or
		" do not " in x or
		" do not " in x or
		" do not " in x or
		" almost " in x
	):
		if (y in ["IV"]):
			return 1.0 / 1
		else:
			return -1.0
	return 0.0

def AllRecipes_Second_Order_Trans_Feature_1(x, y, nx, ny):
	if (";" in x):
		if (ny in ['A-']):
			return 1.0
		else:
			return -1.0
	return 0.0

def AllRecipes_Third_Order_Trans_Feature_1(x, y, nx, ny, nnx, nny):
	if ("." not in x and "." not in nx and y=="A|" and nny=="A|"):
		if (ny in ['A|']):
			return 1.0
		else:
			return -1.0
	return 0.0

def Ifixit_State_Feature_1(x, y):
	if (x.startswith("before ") or
		x.startswith("after ") or
		x.startswith("if ") or
		x.startswith("during ") or
		x.startswith("until ") or
		x.startswith("meanwhile ") or
		x.startswith("when ") or
		x.startswith("while ") or
		x.startswith("repeat ") or
		x.startswith("once ") ):
		if(y in ["N-"]):
			return 1.0
		else:
			return -1.0
	return 0.0

def Ifixit_State_Feature_2(x, y):
	if ("should " in x or
		"etc " in x or
		"do not " in x or
		"don ' t " in x or
		"will " in x or
		"keep in mind " in x or
		" do not " in x or
		" do not " in x or
		" do not " in x or
		" almost " in x
	):
		if (y in ["IV"]):
			return 1.0 / 1
		else:
			return -1.0
	return 0.0

def Ifixit_Second_Order_Trans_Feature_1(x, y, nx, ny):
	if ("." not in x and y=="IV"):
		if (ny in ['IV']):
			return 1.0
		else:
			return -1.0

	if ("." not in x and ny=="IV"):
		if (y in ['IV']):
			return 1.0
		else:
			return -1.0

	return 0.0

def Ifixit_Third_Order_Trans_Feature_1(x, y, nx, ny, nnx, nny):
	if ("." not in x and "." not in nx and y=="IV" and nny=="IV"):
		if (ny in ['IV']):
			return 1.0
		else:
			return -1.0
	return 0.0

def Feature_Triggered_Values(xs, ys):
	state_feature_matrix, trans_feature_matrix = [], []

	if("allrecipes" in pb.dataset):
		state_feature_matrix.append( Get_State_Feature_Vector(xs, ys, AllRecipes_State_Feature_1) )
		state_feature_matrix.append( Get_State_Feature_Vector(xs, ys, AllRecipes_State_Feature_2) )
		trans_feature_matrix.append( Get_Second_Order_Trans_Feature_Vector(xs, ys, AllRecipes_Second_Order_Trans_Feature_1) )
		trans_feature_matrix.append( Get_Third_Order_Trans_Feature_Vector(xs, ys, AllRecipes_Third_Order_Trans_Feature_1) )

	if ("ifixit" in pb.dataset):
		state_feature_matrix.append(Get_State_Feature_Vector(xs, ys, Ifixit_State_Feature_1))
		state_feature_matrix.append(Get_State_Feature_Vector(xs, ys, Ifixit_State_Feature_2))
		trans_feature_matrix.append(Get_Second_Order_Trans_Feature_Vector(xs, ys, Ifixit_Second_Order_Trans_Feature_1))
		trans_feature_matrix.append(Get_Third_Order_Trans_Feature_Vector(xs, ys, Ifixit_Third_Order_Trans_Feature_1))

	return pb.Matrix_Sum(state_feature_matrix), pb.Matrix_Sum(trans_feature_matrix)
