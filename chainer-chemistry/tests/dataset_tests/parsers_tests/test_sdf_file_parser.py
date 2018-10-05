import os

import numpy
import pytest
from rdkit import Chem
import six

from chainer_chemistry.dataset.parsers import SDFFileParser
from chainer_chemistry.dataset.preprocessors import NFPPreprocessor


@pytest.fixture
def mols():
    mol1 = Chem.MolFromSmiles('CN=C=O')
    mol2 = Chem.MolFromSmiles('Cc1ccccc1')
    mol3 = Chem.MolFromSmiles('CC1=CC2CC(CC1)O2')
    return [mol1, mol2, mol3]


@pytest.fixture()
def sdf_file(tmpdir, mols):
    # Chem.AllChem.Compute2DCoords(mol1)
    fname = os.path.join(str(tmpdir), 'test.sdf')
    writer = Chem.SDWriter(fname)
    for mol in mols:
        writer.write(mol)
    return fname


def check_input_features(actual, expect):
    assert len(actual) == len(expect)
    for d, e in six.moves.zip(actual, expect):
        numpy.testing.assert_array_equal(d, e)


def test_sdf_file_parser_not_return_smiles(sdf_file, mols):
    preprocessor = NFPPreprocessor()
    parser = SDFFileParser(preprocessor)
    result = parser.parse(sdf_file, return_smiles=False)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3
    assert smiles is None

    # As we want test SDFFileParser, we assume
    # NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_input_features(dataset[i], expect)


def test_sdf_file_parser_return_smiles(sdf_file, mols):
    preprocessor = NFPPreprocessor()
    parser = SDFFileParser(preprocessor)
    result = parser.parse(sdf_file, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3

    # As we want test SDFFileParser, we assume
    # NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_input_features(dataset[i], expect)

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'Cc1ccccc1'
    assert smiles[2] == 'CC1=CC2CC(CC1)O2'


def test_sdf_file_parser_target_index(sdf_file, mols):
    preprocessor = NFPPreprocessor()
    parser = SDFFileParser(preprocessor)
    result = parser.parse(sdf_file, return_smiles=True, target_index=[0, 2])
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 2

    # As we want test SDFFileParser, we assume
    # NFPPreprocessor works as documented.
    expect = preprocessor.get_input_features(mols[0])
    check_input_features(dataset[0], expect)

    expect = preprocessor.get_input_features(mols[2])
    check_input_features(dataset[1], expect)

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'CC1=CC2CC(CC1)O2'


def test_sdf_file_parser_extract_total_num(sdf_file):
    preprocessor = NFPPreprocessor()
    parser = SDFFileParser(preprocessor)
    num = parser.extract_total_num(sdf_file)
    assert num == 3

# TODO(oono)
# test with non-default options of SDFFileParser


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
