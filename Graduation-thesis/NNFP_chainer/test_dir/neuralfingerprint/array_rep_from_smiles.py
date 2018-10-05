from mol_graph import graph_from_smiles_tuple, degrees
from utils import memoize
import numpy as np

@memoize
def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

'''
Amigdalin,-0.9740000000000001,1,457.4320000000001,7,3,7,202.31999999999996,-0.77,OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O 
Fenfuram,-2.885,1,201.22500000000002,1,2,2,42.24,-3.3,Cc1occc1C(=O)Nc2ccccc2
'''

solubility = -0.77
smiles = ("OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O","Cc1occc1C(=O)Nc2ccccc2")

array = array_rep_from_smiles(smiles)
#print "data length : ", len(array[0])
print array
