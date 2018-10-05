from chainer_chemistry.dataset.preprocessors.common import construct_adj_matrix
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
#from NNFP import mol_graph
from NNFP.Finger_print import array_rep_from_smiles


class NFP_attention_Preprocessor(MolPreprocessor):
    """NFP Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False):
        super(NFP_attention_Preprocessor, self).__init__(add_Hs=add_Hs)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.degree_num = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_adj_matrix(mol, out_size=self.out_size)
        """

		#mol_graph,featureでやってることを書く
        type_check_num_atoms(mol, self.max_atoms)
        smiles = Chem.MolToSmiles(mol)	
        ecfp_array = array_rep_from_smiles(smiles,False)
        fcfp_array = array_rep_from_smiles(smiles,True)
        atom_array1 = ecfp_array['atom_features'].astype(np.float32)
        atom_array2 = fcfp_array['atom_features'].astype(np.float32)
        degree = construct_adj_matrix(mol, out_size=self.out_size)
        self.degree_num = [int(np.sum(x)) for x in degree]
        adj_array = ecfp_array['bond_features']

        #import pdb;pdb.set_trace()
        return atom_array1, atom_array2, adj_array
        #return instances type is "numpy.ndarray"
        #train.py でatom情報とbond情報を取得
    def get_degree(self):
        return self.degree_num
