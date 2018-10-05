import chainer
from chainer import cuda
from chainer import functions as F
from chainer import iterators as I
from chainer import links as L
import numpy as np

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.models import GGNN
from chainer_chemistry.models import MLP
from chainer_chemistry.models import NFP
from chainer_chemistry.models import RSGCN
from chainer_chemistry.models import SchNet
from chainer_chemistry.models import WeaveNet


def build_predictor(method, n_unit, conv_layers, class_num):
    if method == 'nfp':
        print('Use NFP predictor...')
        predictor = GraphConvPredictor(
            NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Use GGNN predictor...')
        predictor = GraphConvPredictor(
            GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Use SchNet predictor...')
        # MLP layer is not necessary for SchNet
        predictor = GraphConvPredictor(
            SchNet(out_dim=class_num, hidden_dim=n_unit, n_layers=conv_layers,
                   readout_hidden_dim=n_unit), None)
    elif method == 'weavenet':
        print('Use WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        predictor = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'rsgcn':
        print('Use RSGCN predictor...')
        predictor = GraphConvPredictor(
            RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'attention':
        print('Use All predictor...')
        predictor = GraphConvPredictor_attention(
            NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid predictor: method={}'.format(method))
    return predictor


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv, mlp=None):
        """Constructor

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer.
                It can be `None` if no operation is necessary after
                `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__(
            attention_layer = L.Linear(16,1)
		)
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        alpha = self.attention_layer(x);
        x = F.batch_matmul(x, alpha)
        if self.mlp:
            x = self.mlp(x)
        return x

    def predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)

class GraphConvPredictor_attention(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv1, graph_conv2, mlp=None):
        """Constructor

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer.
                It can be `None` if no operation is necessary after
                `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv1 = graph_conv1
            self.graph_conv2 = graph_conv2
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x1 = self.graph_conv1(atoms, adjs)
        x2 = self.graph_conv2(atoms, adjs)
        alpha1 = self.attention_layer(x1);
        alpha2 = self.attention_layer(x2);
        x1 = F.batch_matmul(x1,alpha1)
        x2 = F.batch_matmul(x2,alpha2)
        x = x1 + x2
        if self.mlp:
            x = self.mlp(x)
        return x

    def predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)
