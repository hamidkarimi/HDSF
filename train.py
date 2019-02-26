import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import config
import utils
from model import DependencyBLSTM

args = config.args
output_dir = args.project_dir + 'Models/' + args.sim_name + '/'

utils.creat_word_embedding()

def run():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = args.project_dir + 'Models/' + args.sim_name + '/model.pt'

    with open(output_dir + 'config', 'w') as config_file:
        argss = (str(args).split('(')[1].split(')')[0].split(','))
        for a in argss:
            config_file.write("{}\n".format(a))
    if os.path.exists(save_path):
        model = torch.load(save_path)
        model_loaded = True
        print('Great!!! Pre-Trained Model Loaded !!!')
    else:
        model_loaded = False
        print('No pre-trained model ')
        model = DependencyBLSTM(num_words=utils.get_num_words(), max_sen_length=97, max_doc_sent_length=326)

    if not os.path.exists(output_dir + 'train_performance_log.csv'):
        train_performance_log = open(output_dir + 'train_performance_log.csv', 'w')
        train_performance_log.write('Step,Loss\n')
    else:
        train_performance_log = open(output_dir + 'train_performance_log.csv', 'a')

    if not os.path.exists(output_dir + 'eval_performance_log.txt'):
        eval_performance_log = open(output_dir + 'eval_performance_log.txt', 'w')
    else:
        eval_performance_log = open(output_dir + 'eval_performance_log.txt', 'a')

    if args.gpu > -1:
        model.cuda(device=int(args.gpu))

    if not model_loaded:
        if args.fill_embedding:
            embed = utils.get_word_embeddings(source='google')
            if args.gpu > -1:
                model.word_embedding.weight.data.set_(torch.FloatTensor((embed)).cuda(int(args.gpu)))
            else:
                model.word_embedding.weight.data.set_(torch.FloatTensor((embed)))
        else:
            if args.gpu > -1:
                model.word_embedding.weight.data.set_(
                    torch.FloatTensor((np.zeros((utils.get_num_words() + 1, args.word_dim)).astype())).cuda(
                        int(args.gpu)))
            else:
                model.word_embedding.weight.data.set_(
                    torch.FloatTensor((np.zeros((utils.get_num_words() + 1, args.word_dim))).astype(float)))

    if args.train_embeddings == False:
        model.word_embedding.weight.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=params, lr=args.lr, weight_decay=args.l2_coeff)
    model.zero_grad()
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    print('Loading sets...')
    dev_set = utils.get_split_data(split='dev')
    train_set = utils.get_split_data(split='train')
    train_set =train_set[0:int(len(train_set)/2)]
    print('Train and dev sets loaded')

    def train():
        prev_accuracy = 0
        model.train()
        for step in range(args.step_num + 1):
            random.shuffle(train_set)
            docs = train_set[0:args.batch_size]
            labels = Variable(torch.LongTensor([d['label'] for d in docs]))
            doc_encodings, _ = model(docs)
            optimizer.zero_grad()
            outputs = doc_encodings.cpu()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("Step {} Loss {}".format(step, loss.data[0]))
            train_performance_log.write("{},{}\n".format(step, loss.data[0]))
            if step % 20 == 0 and step:
                accuracy = evaluation(step)
                if accuracy >= prev_accuracy:
                    torch.save(model, save_path)
                    print("Best model saved in {} Accuracy {}".format(save_path, accuracy))
                    prev_accuracy = accuracy

    def evaluation(step):
        print('Start evaluation ...')
        model.eval()
        eval_labels = [d['label'] for d in dev_set]
        labels = Variable(torch.LongTensor(eval_labels))
        doc_encodings, _ = model(dev_set)
        outputs = doc_encodings.cpu()
        loss = criterion(outputs, labels).data[0]
        _, predictions = torch.max(outputs.data, 1)
        predictions = predictions.numpy()

        accuracy = accuracy_score(y_true=np.array(eval_labels), y_pred=predictions)
        report = classification_report(y_true=np.array(eval_labels), y_pred=predictions, target_names=['Real', 'Fake'])
        conf_matrix = confusion_matrix(y_true=np.array(eval_labels), y_pred=predictions)
        eval_performance_log.write("Step {}, Loss {} Accuracy {} \n".format(step, loss, accuracy))
        eval_performance_log.write("{}\n".format(report))
        eval_performance_log.write("{}\n".format(conf_matrix))
        eval_performance_log.write("{}\n".format('=' * 50))

        print('************* Evaluation ****************')
        print("Step {}, Loss {}  Accuracy {} ".format(step, loss, accuracy))
        print(report)
        print(conf_matrix)
        print('*****************************************')
        return accuracy

    train()


run()
