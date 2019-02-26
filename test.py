import os
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.autograd import Variable
import config
import utils
from model import DependencyBLSTM

args = config.args
output_dir = args.project_dir + 'Models/' + args.sim_name + '/'
utils.creat_word_embedding()

def test():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = args.project_dir + 'Models/' + args.sim_name + '/model.pt'

    if os.path.exists(save_path):
        model = torch.load(save_path)
        print('Great!!! Pre-Trained Model Loaded !!!')
    else:
        print('No pre-trained model ')
        model = DependencyBLSTM(num_words=utils.get_num_words(), max_sen_length=97, max_doc_sent_length=326)

    if not os.path.exists(output_dir + 'test_performance_log.txt'):
        test_performance_log = open(output_dir + 'test_performance_log.txt', 'w')
    else:
        test_performance_log = open(output_dir + 'test_performance_log.txt', 'a')

    if args.gpu > -1:
        model.cuda(device=int(args.gpu))
    if args.fill_embedding :
        embed = utils.get_word_embeddings(source='google')
        if args.gpu > -1:
            model.word_embedding.weight.data.set_(torch.FloatTensor((embed)).cuda(int(args.gpu)))
        else:
            model.word_embedding.weight.data.set_(torch.FloatTensor((embed)))
    else:
        if args.gpu > -1:
            model.word_embedding.weight.data.set_(
                torch.FloatTensor((np.zeros((utils.get_num_words() + 1, args.word_dim)).astype())).cuda(int(args.gpu)))
        else:
            model.word_embedding.weight.data.set_(
                torch.FloatTensor((np.zeros((utils.get_num_words() + 1, args.word_dim))).astype(float)))

    if args.train_embeddings == False:
        model.word_embedding.weight.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    test_set = utils.get_split_data(split='test')
    print('Start Test ...')
    model.eval()
    test_labels = [d['label'] for d in test_set]
    labels = Variable(torch.LongTensor(test_labels))

    lengths = [len(doc['word_indices']) for doc in test_set]
    doc_encodings, all_doc_doc_dependency_tree_info = model(test_set)

    outputs = doc_encodings.cpu()
    loss = criterion(outputs, labels).data[0]
    _, predictions = torch.max(outputs.data, 1)
    predictions = predictions.numpy()

    with open(output_dir + "matrix.pkl", 'wb') as f:
        pickle.dump([lengths, all_doc_doc_dependency_tree_info, test_labels], f)

    accuracy = accuracy_score(y_true=np.array(test_labels), y_pred=predictions)
    report = classification_report(y_true=np.array(test_labels), y_pred=predictions, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(y_true=np.array(test_labels), y_pred=predictions)
    test_performance_log.write(" Loss {} Accuracy {} \n".format(loss, accuracy))
    test_performance_log.write("{}\n".format(report))
    test_performance_log.write("{}\n".format(conf_matrix))
    test_performance_log.write("{}\n".format('=' * 50))

    print('************* Test ****************')
    print("Loss {}  Accuracy {} ".format(loss, accuracy))
    print(report)
    print(conf_matrix)
    print('*****************************************')
    return accuracy


test()

fake_doc_stat, real_doc_stat = utils.dependecy_tree_stat(dir=output_dir)
#print(fake_doc_stat)
#print('*' * 100)
#print(real_doc_stat)
#print('*' * 100)
print("Avg. Number of Leaf Nodes: Fake {} Real {}".format(np.mean(fake_doc_stat[:, 2]), np.mean(real_doc_stat[:, 2])))
print("Avg. Preorder Difference: Fake {} Real {}".format(np.mean(fake_doc_stat[:, 4]), np.mean(real_doc_stat[:, 4])))
print("Avg. Parent-Child Distance: Fake {} Real {}".format(np.mean(fake_doc_stat[:, 3]), np.mean(real_doc_stat[:, 3])))
