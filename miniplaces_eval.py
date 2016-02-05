from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, ResidualModule, Activation
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--network', default='plain', choices=['plain', 'resnet'],
                    help='type of network to create (plain or resnet)')
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=112, scale_range=140, repo_dir=args.data_dir)

#train = ImageLoader(set_name='train', shuffle=True, do_transforms=True,
#                   inner_size=112, scale_range=(128,240), repo_dir=args.data_dir)



def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=True)


def module_factory(nfm, stride=1):
    projection = None if stride == 1 else IdentityInit()
    module = [Conv(**conv_params(3, nfm, stride=stride)),
              Conv(**conv_params(3, nfm, relu=False))]
    module = module if args.network == 'plain' else [ResidualModule(module, projection)]
    module.append(Activation(Rectlin()))
    return module


# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 5) for stage in sorted(range(4) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
from neon.layers import ColorNoise
#layers = [ColorNoise()]

# layers.append(Affine(nout=100, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

scales = [112, 128, 160, 240]

for scale in scales:
    print scale

    layers = []
    layers += [Conv(**conv_params(7, 32, 2))]
    for nfm, stride in zip(nfms, strides):
        layers.append(module_factory(nfm, stride))
    layers.append(Pooling(7, op='avg'))

    layers.append(Conv(fshape=(1,1,100), init=Kaiming(local=True), batch_norm=True))
    layers.append(Pooling(fshape='all', op='avg'))
    layers.append(Activation(Softmax()))

    model = Model(layers=layers)
    test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, inner_size=scale,
                       scale_range=scale, repo_dir=args.data_dir)

    model.load_params("/home/users/hunter/bigfeat_dropout.pkl")

    softmaxes = model.get_outputs(test)
    from neon.util.persist import save_obj
    save_obj(softmaxes, "bigfeat_dropout_SM_{}.pkl".format(scale))
