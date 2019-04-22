import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def path(p):
    return os.path.expanduser(p)


parser = argparse.ArgumentParser(description='Arguments for HDSF Project')

parser.add_argument("--project_dir", type=str, required=False, default='/PATH/DHSF/', help="project directory")
parser.add_argument("--dropout", type=float, required=False, default=0.0, help="Value of droupout")
parser.add_argument("--sim_name", type=str, required=False, default="sim1",help="The unique and arbitrary name of simulation")
parser.add_argument("--l2_coeff", type=float, required=False, default=0.00,help="Value L2 regularization coefficient")
parser.add_argument("--train_embeddings", type=str2bool, required=False, default=True,help="train embeddings or not")
parser.add_argument("--lr", type=float, required=False, default=0.01, help="initial learning rate ")
parser.add_argument("--batch_size", type=int, required=False, default=40, help="size of batch")
parser.add_argument('--step_num', type=int, required=False, help=' number of steps to run the simulation.', default=200)
parser.add_argument('--gpu', default=-1, type=int, help='GPU id > 0 moves the model on GPU')
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument("--fill_embedding", type=str2bool, required=False, default=True, help="train embeddings or not")
parser.add_argument('--blstm_hidden_unit_dim', default=100, type=int)

args = parser.parse_args()
