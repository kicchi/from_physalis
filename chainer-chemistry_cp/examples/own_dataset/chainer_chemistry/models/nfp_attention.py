
"""
Implementation of Neural functionsingerprint with attention layer

"""
import chainer
from chainer import functions
from chainer import links
from chainer import Variable
import numpy

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry import links
from NNFP import Finger_print

class NFPUpdate(chainer.Chain):
    """NFP sub module for update part

    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
        max_degree (int): max degree of edge
    """

    def __init__(self, in_channels, out_channels, max_degree=6):
        super(NFPUpdate, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.graph_linears = chainer.ChainList(
                *[links.GraphLinear(in_channels, out_channels)
                  for _ in range(num_degree_type)]
            )
        self.max_degree = max_degree
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj, deg_conds):
        # h    (minibatch, atom, ch)
        # h encodes each atom's info in ch axis of size hidden_dim
        # adjs (minibatch, atom, atom)

        # --- Message part ---
        # Take sum along adjacent atoms

        # fv   (minibatch, atom, ch)
        fv = chainer_chemistry.functions.matmul(adj, h)

        # --- Update part ---
        # s0, s1, s2 = fv.shape
        if self.xp is numpy:
            zero_array = numpy.zeros(fv.shape, dtype=numpy.float32)
        else:
            zero_array = self.xp.zeros_like(fv)

        fvds = [functions.where(cond, fv, zero_array) for cond in deg_conds]

        out_h = 0
        for graph_linear, fvd in zip(self.graph_linears, fvds):
            out_h = out_h + graph_linear(fvd)

        # out_x shape (minibatch, max_num_atoms, hidden_dim)
        out_h = functions.sigmoid(out_h)
        return out_h


class NFPReadout(chainer.Chain):
    """NFP sub module for readout part

    Args:
        in_channels (int): dimension of feature vector associated to each
            atom (node)
        out_size (int): output dimension of feature vector associated to each
            molecule (graph)
    """

    def __init__(self, in_channels, out_size):
        super(NFPReadout, self).__init__()
        with self.init_scope():
            self.output_weight = chainer_chemistry.links.GraphLinear(
                in_channels, out_size)
        self.in_channels = in_channels
        self.out_size = out_size

    def __call__(self, h):
        # input  h shape (minibatch, atom, ch)
        # return i shape (minibatch, ch)

        # --- Readout part ---
        i = self.output_weight(h)
        i = functions.softmax(i, axis=2)  # softmax along channel axis
        i = functions.sum(i, axis=1)  # sum along atom's axis
        return i

class NFP_attention(chainer.Chain):
	"""Neural functionsinger Print (NfunctionsP)
	Args:
		out_dim (int): dimension of output feature vector
		hidden_dim (int): dimension of feature vector
     		associated to each atom
		max_degree (int): max degree of atoms
		    when molecules are regarded as graphs
		n_atom_types (int): number of types of atoms
		n_layer (int): number of layers
	"""

	def __init__(self, out_dim, hidden_dim=16, n_layers=4, max_degree=6,
					n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False, model_params=None):
		super(NFP_attention, self).__init__() 
		num_degree_type = max_degree + 1 
		with self.init_scope():
			#self.embed = chainer_chemistry.links.EmbedAtomID(
			#	in_size=n_atom_types, out_size=hidden_dim)
			#self.layers = chainer.ChainList(
			#	*[NFPUpdate(hidden_dim, hidden_dim, max_degree=max_degree)
			# 	  for _ in range(n_layers)])
			#self.read_out_layers = chainer.ChainList(
			#	*[NFPReadout(hidden_dim, out_dim)
		   	#	  for _ in range(n_layers)])
			self.attention_layer_1 = chainer.links.Linear(out_dim,1)
			self.out_dim = out_dim
			self.hidden_dim = hidden_dim
			self.max_degree = max_degree
			self.num_degree_type = num_degree_type
			self.n_layers = n_layers
			self.concat_hidden = concat_hidden
			self.build_fp1 = Finger_print.ECFP(model_params)
			self.build_fp2 = Finger_print.FCFP(model_params)

	def __call__(self, atom_array1, atom_array2, adj, degree_num):
	
		"""functionsorward propagation
		Args:
			atom_array (numpy.ndarray): minibatch of molecular which is
				represented with atom IDs (representing C, O, S, ...)
				`atom_array[mol_index, atom_index]` represents `mol_index`-th
				molecule's `atom_index`-th atomic number
			adj (numpy.ndarray): minibatch of adjancency matrix
				`adj[mol_index]` represents `mol_index`-th molecule's
				adjacency matrix
	
		Returns:
			~chainer.Variable: minibatch of fingerprint
		"""
		#Finger_print でやっていることを書く
		def concat_3_to_2(array):
			ret = numpy.empty((0,len(array[0][0])), numpy.float32)
			for arr in array:
				ret = numpy.concatenate([ret,arr])
			return ret
		atom_array1 = concat_3_to_2(atom_array1)
		atom_array2 = concat_3_to_2(atom_array2)

		fp1 = self.build_fp1(atom_array1, adj, degree_num)
		import pdb;pdb.set_trace()
		fp2 = self.build_fp2(atom_array2, adj, degree_num)

		fp1_alpha = self.attention_layer_1(fp1)
		fp2_alpha = self.attention_layer_2(fp2)

		fp1_alpha = functions.softmax(fp1_alpha)
		fp2_alpha = functions.softmax(fp2_alpha)

		attention_fp1 = functions.batchmatmul(fp1,fp1_alpha)
		attention_fp2 = functions.batchmatmul(fp2,fp2_alpha)

		g = attention_fp1 + attention_fp2

		if self.concat_hidden:
			return functions.concat(g_list, axis=2)
		else:
			return g

