#coding: utf-8
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from rdkit import Chem


def result_plot(curve, model_params):
	plt.title("1 dim label")	
	x = np.linspace(0, 100, 10)
	y = curve
	plt.plot(x,y, label='loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()

def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def smiles_from_SDF(filename, sizes):
	smiles_list = np.empty((1,1))
	target_list = np.empty((1,1))
	inf = open(filename,'r')	
	mol_data =""
	while 1:
		line = inf.readline()
		if not line:
			break
		if line[:4] == "$$$$":
			mol = Chem.MolFromMolBlock(mol_data)
			#print str(Chem.MolFromMolBlock(mol_data))
			smiles_list = np.append(smiles_list, [[str(Chem.MolToSmiles(mol))]], axis=0)
			mol_data = ""
		elif line.find("ctive"):
			if line[0] == 'A':
				target_list = np.append(target_list, [[1]], axis=0)
			elif line[0] == 'I':
			 	target_list = np.append(target_list, [[0]], axis=0) 
			mol_data = mol_data+line
		else:
			mol_data = mol_data+line

	slices = []
	start = 1
	for size in sizes:
		stop = start + size 
		slices.append(slice(start, stop)) 
		start = stop

	stops = [s.stop for s in slices]
	if not all(stops):
		raise Exception("Slices can't be open-ended")

	return [(smiles_list[s], target_list[s]) for s in slices]


def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]


def list_concat(lists):
    return list(it.chain(*lists))
    
def load_data_slices(filename, slice_lists, input_name, target_name):
    stops = [s.stop for s in list_concat(slice_lists)]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)

    return [(np.concatenate([data[0][s] for s in slices], axis=0),
             np.concatenate([data[1][s] for s in slices], axis=0))
            for slices in slice_lists]

def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)

def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)

def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))

def data_dir():
    return os.path.expanduser(safe_get("DATA_DIR"))

def safe_get(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception("%s environment variable not set" % varname)
