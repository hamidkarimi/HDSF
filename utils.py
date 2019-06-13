import pickle
import numpy as np
import pandas as pd
from torch.autograd import Variable
import config
import gensim.models.keyedvectors as word2vec
import os
import urllib.request
args = config.args


###################### Tree ################################################
class myTree(object):
    def __init__(self, name='Node', children=None, data=None, parent=None):
        self.parent = parent
        self.name = name
        self.index = -1
        self.children = []
        self.data = data
        self.characters = []
        self.parent_relation_index = -1
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, myTree)
        self.children.append(node)

    def __str__(self):
        if len(self.children) == 0:
            x = 'NONE'
        else:
            x = [a.name for a in self.children]
        return "{} Children:{}".format(self.name, x)

    def __getitem__(self, item):
        return self.name


def get_leaves(node):
    leaves = []
    if len(node.children) == 0:
        leaves.append(node)
    else:
        for child in node.children:
            leaves.extend(get_leaves(child))
    return leaves


def get_preorder(tree, X):
    X.append((tree.name))
    if len(tree.children) == 0:
        return
    for t in tree.children:
        X.append(get_preorder(t, X))


###################### Utility Functions  ################################################

def wrap_with_variable(tensor, gpu, requires_grad=True):
    if gpu > -1:
        return Variable(tensor.cuda(gpu), requires_grad=requires_grad)
    else:
        return Variable(tensor, requires_grad=requires_grad)


def get_word_embeddings(source='google'):
    with open(args.project_dir + 'Data/word_emb_' + source + '.pkl', 'rb') as f:
        embed = pickle.load(f)
    return embed


def creat_word_embedding():
    if os.path.exists(args.project_dir + 'Data/word_emb_google.pkl'):
        return
    if not os.path.exists(args.project_dir + 'GoogleNews-vectors-negative300.bin'):
        print("Please download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and place it in".format(args.project_dir))
        exit(-1)
    print('Creating word embeddings ...')
    model = word2vec.KeyedVectors.load_word2vec_format(args.project_dir + 'GoogleNews-vectors-negative300.bin',
                                                       binary=True)
    all_words = pd.read_csv(args.project_dir + "Data/words.csv", names=['Index', 'Word', 'Freq'])
    word_embeddings = np.zeros(shape=(len(all_words) + 1, args.word_dim))
    non_exist_words = []
    for w in all_words.values:
        if (w[1] in model):
            word_embeddings[w[0]] = model[w[1]]
        else:
            word_embeddings[w[0]] = np.random.rand(1, args.word_dim)[0]
            non_exist_words.append(w[1])
    word_embeddings[len(all_words)] = np.random.rand(1, args.word_dim)[0]
    with open(args.project_dir + 'Data/word_emb_google.pkl', 'wb') as pkl:
        pickle.dump(word_embeddings, pkl)
    print('Word embedding file created and save in {}'.format(args.project_dir + 'Data/word_emb_google.pkl'))
    #print(non_exist_words, len(non_exist_words))


def get_split_data(split='train'):
    documents = []
    with open(args.project_dir + 'Splits/' + split + '.split.csv', 'r') as f:
        records = f.read().splitlines()

    for record in records:
        label = record.split(',')[-1]
        if label == 'Fake':
            l = 1
        else:
            l = 0
        with open(args.project_dir + 'Data/' + record.split(',')[1], 'r') as f:
            word_indices = f.read().splitlines()
            word_indices = [w.split(',') for w in word_indices]
            documents.append({"word_indices": word_indices, 'label': l})
    return documents


def get_num_words():
    all_words = pd.read_csv(args.project_dir + "Data/words.csv", names=['Index', 'Word', 'Freq'])
    return len(all_words)


def construct_dependecy_tree(matrixp, rootp):
    trees = []
    root_index = np.argmax(rootp)
    root = myTree(name=str(root_index))
    trees.append(root)
    current_nodes = []
    current_nodes.append(root_index)
    matrixp[:, root_index] = np.array([-1 for _ in range(len(rootp))])
    flag = True
    child_parent_diff = 0
    while flag:
        currmax = -99
        father = -1
        child = -1
        for c in current_nodes:
            m = np.argmax(matrixp[c, :])
            if matrixp[c, m] > currmax:
                currmax = matrixp[c, m]
                father = c
                child = m

        father_index_in_the_list = -1
        for i, t in enumerate(trees):
            if t.name == str(father):
                father_index_in_the_list = i
                break
        node = myTree(name=str(child))
        child_parent_diff += np.abs(int(child) - int(trees[father_index_in_the_list].name))
        node.parent = trees[father_index_in_the_list]
        trees[father_index_in_the_list].add_child(node)
        trees[father_index_in_the_list].data = currmax
        trees.append(node)
        current_nodes.append(child)
        matrixp[:, child] = np.array([-1 for _ in range(len(rootp))])
        if len(current_nodes) == len(rootp):
            flag = False
    return trees[0], child_parent_diff


def dependecy_tree_stat(dir):
    with open(dir + 'matrix.pkl', 'rb') as f:
        x = pickle.load(f)
        lengths, all_doc_doc_dependency_tree_info, labels = x[0], x[1], x[2]
        pijs = [a[0].numpy() for a in all_doc_doc_dependency_tree_info]
        piroots = [a[1].numpy() for a in all_doc_doc_dependency_tree_info]
    fake_doc_stat = []
    real_doc_stat = []
    for index, (matrixp, rootp, label) in enumerate(zip(pijs, piroots, labels)):
        tree, child_parent_diff = construct_dependecy_tree(matrixp, rootp)
        X = []
        get_preorder(tree, X)
        Y = []
        for ss in X:
            if ss is not None:
                Y.append(int(ss))
        preorder_diff = 0
        for i, n in enumerate(Y):
            preorder_diff += np.abs(i + 1 - n)
        leaves = get_leaves(tree)
        if label == 1:
            fake_doc_stat.append([label, len(matrixp), len(leaves) / (np.log10(len(matrixp))),
                                  child_parent_diff / (np.log10(len(matrixp))),
                                  preorder_diff / (np.log10(len(matrixp)))])
        else:
            real_doc_stat.append([label, len(matrixp), len(leaves) / (np.log10(len(matrixp))),
                                  child_parent_diff / (np.log10(len(matrixp))),
                                  preorder_diff / (np.log10(len(matrixp)))])

    fake_doc_stat = np.array(fake_doc_stat)
    real_doc_stat = np.array(real_doc_stat)
    return fake_doc_stat, real_doc_stat

#creat_word_embedding()
