import numpy as np
import torch
import torch.nn as nn
import config
import utils
args = config.args
class DependencyBLSTM(nn.Module):
    def __init__(self, num_words, max_sen_length, max_doc_sent_length):
        super(DependencyBLSTM, self).__init__()
        self.max_sen_length = max_sen_length
        self.max_doc_sent_length = max_doc_sent_length
        self.dropout = nn.Dropout(p=args.dropout)
        self.word_embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=args.word_dim)
        self.Softmax = nn.Softmax(dim=0)
        ############################# Sentence level Functions #######################################

        self.forwardLSTM_sent = nn.LSTM(num_layers=1, input_size=args.word_dim,
                                        dropout=args.dropout,
                                        hidden_size=int(args.blstm_hidden_unit_dim),
                                        batch_first=True)
        self.backwardLSTM_sent = nn.LSTM(num_layers=1, input_size=args.word_dim,
                                         dropout=args.dropout,
                                         hidden_size=args.blstm_hidden_unit_dim,
                                         batch_first=True)
        self.sentence_encoder = nn.Sequential(nn.Linear(args.word_dim, args.blstm_hidden_unit_dim),
                                              nn.LeakyReLU(), nn.Dropout(p=args.dropout))
        ############################# Doc level Functions #######################################

        self.parent_encoder_doc = nn.Sequential(
            nn.Linear(args.blstm_hidden_unit_dim, int(args.blstm_hidden_unit_dim)),
            nn.LeakyReLU(), nn.Dropout(p=args.dropout))
        self.child_encoder_doc = nn.Sequential(
            nn.Linear(args.blstm_hidden_unit_dim, int(args.blstm_hidden_unit_dim)),
            nn.LeakyReLU(), nn.Dropout(p=args.dropout))


        self.root_score_encoder_doc = nn.Linear(args.blstm_hidden_unit_dim, 1)
        self.root_embed_doc = \
            utils.wrap_with_variable(torch.FloatTensor(np.zeros(shape=(args.blstm_hidden_unit_dim))),
                                     gpu=args.gpu,
                                     requires_grad=True)
        self.r_embeds_doc = nn.Sequential(
            nn.Linear(3 * args.blstm_hidden_unit_dim,
                      int(args.blstm_hidden_unit_dim)),
            nn.LeakyReLU(), nn.Dropout(p=args.dropout))

        self.final_binary_classifier = nn.Linear(int(args.blstm_hidden_unit_dim), 2)

    def create_sentence_batches(self, docs):
        all_doc_batches = []
        all_doc_batches_inverse = []
        all_doc_seq_lengths = []
        for doc in docs:
            doc_sent_embed, doc_sent_embed_inverse = [], []
            seq_lengths = []
            for sent_word_indices in doc['word_indices']:
                j = utils.wrap_with_variable(torch.LongTensor(np.array(sent_word_indices).astype(int)), gpu=args.gpu, requires_grad=False)

                word_embed = self.word_embedding(j)
                word_embed = self.dropout(word_embed)
                X = torch.zeros(self.max_sen_length, args.word_dim)
                X[0:len(sent_word_indices)] = word_embed.data
                X = utils.wrap_with_variable(X, gpu=args.gpu, requires_grad=True)
                doc_sent_embed.append(X)

                idx = [i for i in range(word_embed.data.size(0) - 1, -1, -1)]
                if args.gpu > -1:
                    idx = torch.LongTensor(idx).cuda(args.gpu)
                else:
                    idx = torch.LongTensor(idx)
                X_inverse = torch.zeros(self.max_sen_length, args.word_dim)
                X_inverse[0:len(sent_word_indices)] = word_embed.data.index_select(0, idx)
                X_inverse = utils.wrap_with_variable(X_inverse, gpu=args.gpu, requires_grad=True)
                doc_sent_embed_inverse.append(X_inverse)

                seq_lengths.append(len(sent_word_indices))

            doc_sent_embed = torch.stack(doc_sent_embed)
            doc_sent_embed_inverse = torch.stack(doc_sent_embed_inverse)
            all_doc_batches.append(doc_sent_embed)
            all_doc_batches_inverse.append(doc_sent_embed_inverse)
            all_doc_seq_lengths.append(seq_lengths)
        return all_doc_batches, all_doc_batches_inverse, all_doc_seq_lengths

    def get_sentence_encodings(self, all_doc_batches, all_doc_batches_inverse, all_doc_seq_lengths):
        all_doc_sentence_encodings = []
        for doc_batch, doc_batch_inverse, doc_seq_length in zip(all_doc_batches, all_doc_batches_inverse,
                                                                all_doc_seq_lengths):
            doc_sentence_encodings = []

            fwrd_outputs, _ = self.forwardLSTM_sent(doc_batch)
            bwrd_outputs, _ = self.backwardLSTM_sent(doc_batch_inverse)
            for sent_forward, sent_backward, l in zip(doc_batch, doc_batch_inverse, doc_seq_length):
                idx = [i for i in range(l - 1, -1, -1)]
                if args.gpu > -1:
                    idx = torch.LongTensor(idx).cuda(args.gpu)
                else:
                    idx = torch.LongTensor(idx)
                bwrd_outputs_inverse = utils.wrap_with_variable(sent_backward.data.index_select(0, idx), gpu=args.gpu,
                                                                requires_grad=True)

                h = self.sentence_encoder(0.5 * (sent_forward[l - 1] + bwrd_outputs_inverse[l-1]))
                doc_sentence_encodings.append(h)
            doc_sentence_encodings = torch.stack(doc_sentence_encodings)
            all_doc_sentence_encodings.append(doc_sentence_encodings)
        return all_doc_sentence_encodings

    def forward(self, docs):
        all_doc_batches, all_doc_batches_inverse, all_doc_seq_lengths = self.create_sentence_batches(docs)
        all_doc_sentence_encodings = self.get_sentence_encodings(all_doc_batches, all_doc_batches_inverse,
                                                                 all_doc_seq_lengths)
        all_doc_doc_dependency_tree_info = []
        all_final_features = []
        for sentence_encodings in all_doc_sentence_encodings:
            fri = []
            Aij = []
            for i in range(len(sentence_encodings)):
                fri.append(self.root_score_encoder_doc(sentence_encodings[i]))
                for j in range(len(sentence_encodings)):
                    if i == j:
                        Aij.append(utils.wrap_with_variable(torch.tensor(-9999999.000), gpu=args.gpu, requires_grad=True))
                        continue

                    x = torch.dot(self.parent_encoder_doc(sentence_encodings[i]),
                                  self.child_encoder_doc(sentence_encodings[j]))
                    Aij.append(x)
            Aij = torch.stack(Aij)
            Aij = Aij.view(len(sentence_encodings), len(sentence_encodings))
            Aij = self.Softmax(Aij)
            fri = torch.stack(fri)
            fri = self.Softmax(fri)
            ri = []
            for i in range(len(sentence_encodings)):
                tmp = []
                tmp3 = []
                for k in range(len(sentence_encodings)):
                    tmp.append(torch.mul(Aij[k, i], sentence_encodings[k]))
                    tmp3.append(torch.mul(Aij[i, k], sentence_encodings[i]))
                tmp3 = torch.stack(tmp3)
                tmp3 = torch.sum(tmp3, 0)
                tmp = torch.stack(tmp)
                tmp = torch.sum(tmp, 0)
                tmp2 = torch.mul(fri[i], self.root_embed_doc)
                ri.append(self.r_embeds_doc(torch.cat((sentence_encodings[i], tmp + tmp2, tmp3))))
            ri = torch.stack(ri)
            final_feature = torch.mean(ri,0)
            all_final_features.append(final_feature)
            all_doc_doc_dependency_tree_info.append([Aij.data, fri.data])

        all_final_features = torch.stack(all_final_features)
        output = self.final_binary_classifier(all_final_features)
        return output, all_doc_doc_dependency_tree_info
