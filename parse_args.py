import argparse
from os.path import join


def get_basic_cl_args(parser):
    # CL loss args
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--num-pos', type=int, default=1)
    parser.add_argument('--use-fc-head', type=bool, default=True)
    parser.add_argument('--num-rand-negs', type=int, default=64)
    parser.add_argument('--class-negs', type=bool, required=True,
                        help='if True takes negative examples only from the same class and ' +
                             'limits the number of classes in each batch')
    # used only if class negs is True
    parser.add_argument('--num-b-cls-samp', type=int, required=True,
                        help="number of samples from each class in each batch")
    parser.add_argument('--num-b-cls', type=int, required=True,
                        help="number of classes in each batch")

    return parser


def get_general_args(parser):

    # region Paths & Names args
    parser.add_argument('-bd', '--base-dir', type=str, required=True,
                        help="a directory in which all models checkpoints and tensorbaord files are saved")
    parser.add_argument('--exp-name', type=str, required=True, help='experiment name')
    parser.add_argument('-dn', '--data-name', type=str, required=True,
                        help="""
                        options are: 
                        || cars3d_train || cars3d_test ||
                        || smallnorb_train || smallnorb_test ||
                        || celeba_x64_train || celeba_x64_test ||
                        || edges2shoes_x64_train || edges2shoes_x64_test ||
                        || shapes3d__class_shape__train || shapes3d__class_shape__test ||  
                        """)
    parser.add_argument('--test-data-name', type=str, default=None,
                        help="""
                        options are: 
                        || cars3d_train || cars3d_test ||
                        || smallnorb_train || smallnorb_test ||
                        || celeba_x64_train || celeba_x64_test ||
                        || edges2shoes_x64_train || edges2shoes_x64_test ||
                        || shapes3d__class_shape__train || shapes3d__class_shape__test ||  
                        """)

    # endregion

    # region load weights + starting epoch
    parser.add_argument('--load-weights', type=bool, default=False)
    parser.add_argument('--load-weights-exp', type=str, default=None)
    parser.add_argument('--load-weights-epoch', type=str, default='last')
    # endregion load weights + starting epoch

    # region Algorithmic Global Params
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--content-dim', type=int, required=True)
    parser.add_argument('--class-dim', type=int, required=True)
    parser.add_argument('--used-transforms', type=str, default=None)
    parser.add_argument('--use-pretrain', type=bool, default=True)

    # endregion Algorithmic Global Params

    # region CPU + GPU
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--num-workers", default=4, type=int, help="dataloader num_workers")
    # endregion CPU + GPU

    # region architecture
    parser.add_argument('--enc-arch', type=str, required=True,
                        help="moco_resnet50 || None")
    # endregion architecture

    return parser