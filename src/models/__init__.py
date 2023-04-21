from .srcnet import SRCNet
from .hlnet import HLNet
from .scinet import SCINet
from .cnn import CNN
from .nbeats import NBeatsNet
from .fc import FullyConnected
from .lstm import LSTM
import inspect

models = {
    'srcnet': SRCNet,
    'cnn': CNN,
    'scinet': SCINet,
    'nbeats': NBeatsNet,
    'hlnet': HLNet,
    'lstm': LSTM,
    'fc': FullyConnected
}
def get_model(args):


    model_class = models[args['model'].lower()]

    model_inspect = inspect.getfullargspec(model_class)
    arguments = model_inspect.args[1:]
    kargs = {arg: args.get(arg) for arg in arguments}
    print(kargs)
    return model_class(**kargs).double()