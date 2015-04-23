# sequence to sequence learning
# LSTM without peepholes
# Adagrad
# All words of target sentence conditioned on hidden vector of source sentence
from __future__ import division
import os
import time
import json
from collections import defaultdict
from collections import OrderedDict
from scipy import stats
import random
import numpy
import theano
from theano import tensor as T

class RNNLM(object):
    """recurrent neural network language model"""
    def __init__(self, nh, nw):
        """
        nh :: dimension of the hidden layer
        nw :: vocabulary size
        """
        # parameters of the model
        self.index = theano.shared(name='index',
                                value=numpy.eye(nw,
                                dtype=theano.config.floatX))
        # parameters of the first LSTM
        self.wxg_1 = theano.shared(name='wxg_1',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whg_1 = theano.shared(name='whg_1',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxi_1 = theano.shared(name='wxi_1',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whi_1 = theano.shared(name='whi_1',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxf_1 = theano.shared(name='wxf_1',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whf_1 = theano.shared(name='whf_1',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxo_1 = theano.shared(name='wxo_1',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.who_1 = theano.shared(name='who_1',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.bg_1 = theano.shared(name='bg_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi_1 = theano.shared(name='bi_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf_1 = theano.shared(name='bf_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo_1 = theano.shared(name='bo_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0_1 = theano.shared(name='h0_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0_1 = theano.shared(name='c0_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.wxg_acc_1 = theano.shared(name='wxg_acc_1',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whg_acc_1 = theano.shared(name='whg_acc_1',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxi_acc_1 = theano.shared(name='wxi_acc_1',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whi_acc_1 = theano.shared(name='whi_acc_1',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxf_acc_1 = theano.shared(name='wxf_acc_1',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whf_acc_1 = theano.shared(name='whf_acc_1',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxo_acc_1 = theano.shared(name='wxo_acc_1',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.who_acc_1 = theano.shared(name='who_acc_1',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.bg_acc_1 = theano.shared(name='bg_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi_acc_1 = theano.shared(name='bi_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf_acc_1 = theano.shared(name='bf_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo_acc_1 = theano.shared(name='bo_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0_acc_1 = theano.shared(name='h0_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0_acc_1 = theano.shared(name='c0_acc_1',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # parameters of the second LSTM
        self.wxg_2 = theano.shared(name='wxg_2',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whg_2 = theano.shared(name='whg_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wh_1g_2 = theano.shared(name='wh_1g_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxi_2 = theano.shared(name='wxi_2',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whi_2 = theano.shared(name='whi_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wh_1i_2 = theano.shared(name='wh_1i_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxf_2 = theano.shared(name='wxf_2',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.whf_2 = theano.shared(name='whf_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wh_1f_2 = theano.shared(name='wh_1f_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wxo_2 = theano.shared(name='wxo_2',
                                value=0.02 * numpy.random.randn(nw, nh)
                                .astype(theano.config.floatX))
        self.who_2 = theano.shared(name='who_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.wh_1o_2 = theano.shared(name='wh_1o_2',
                                value=0.02 * numpy.random.randn(nh, nh)
                                .astype(theano.config.floatX))
        self.w_2 = theano.shared(name='w_2',
                                value=0.02 * numpy.random.randn(nh, nw)
                               .astype(theano.config.floatX))
        self.bg_2 = theano.shared(name='bg_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi_2 = theano.shared(name='bi_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf_2 = theano.shared(name='bf_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo_2 = theano.shared(name='bo_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b_2 = theano.shared(name='b_2',
                               value=numpy.zeros(nw,
                               dtype=theano.config.floatX))

        self.wxg_acc_2 = theano.shared(name='wxg_acc_2',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whg_acc_2 = theano.shared(name='whg_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wh_1g_acc_2 = theano.shared(name='wh_1g_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxi_acc_2 = theano.shared(name='wxi_acc_2',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whi_acc_2 = theano.shared(name='whi_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wh_1i_acc_2 = theano.shared(name='wh_1i_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxf_acc_2 = theano.shared(name='wxf_acc_2',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whf_acc_2 = theano.shared(name='whf_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wh_1f_acc_2 = theano.shared(name='wh_1f_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxo_acc_2 = theano.shared(name='wxo_acc_2',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.who_acc_2 = theano.shared(name='who_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wh_1o_acc_2 = theano.shared(name='wh_1o_acc_2',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.w_acc_2 = theano.shared(name='w_acc_2',
                                value=numpy.zeros((nh, nw),
                                dtype=theano.config.floatX))
        self.bg_acc_2 = theano.shared(name='bg_acc_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi_acc_2 = theano.shared(name='bi_acc_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf_acc_2 = theano.shared(name='bf_acc_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo_acc_2 = theano.shared(name='bo_acc_2',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b_acc_2 = theano.shared(name='b_acc_2',
                               value=numpy.zeros(nw,
                               dtype=theano.config.floatX))

        #bundle
        self.params = [self.wxg_1, self.whg_1, self.wxi_1, self.whi_1,
                self.wxf_1, self.whf_1, self.wxo_1, self.who_1, self.bg_1,
                self.bi_1, self.bf_1, self.bo_1, self.h0_1, self.c0_1,
                self.wxg_2, self.whg_2, self.wh_1g_2, self.wxi_2, self.whi_2,
                self.wh_1i_2, self.wxf_2,
                self.whf_2, self.wh_1f_2, self.wxo_2, self.who_2, self.wh_1o_2, self.w_2, self.bg_2,
                self.bi_2, self.bf_2, self.bo_2, self.b_2]
        self.params_acc = [self.wxg_acc_1, self.whg_acc_1, self.wxi_acc_1,
                self.whi_acc_1, self.wxf_acc_1, self.whf_acc_1, self.wxo_acc_1,
                self.who_acc_1, self.bg_acc_1, self.bi_acc_1, self.bf_acc_1,
                self.bo_acc_1, self.h0_acc_1, self.c0_acc_1, self.wxg_acc_2, 
                self.whg_acc_2, self.wh_1g_acc_2, self.wxi_acc_2,
                self.whi_acc_2, self.wh_1i_acc_2, self.wxf_acc_2,
                self.whf_acc_2, self.wh_1f_acc_2, self.wxo_acc_2,
                self.who_acc_2, self.wh_1o_acc_2, self.w_acc_2,
                self.bg_acc_2, self.bi_acc_2, self.bf_acc_2, self.bo_acc_2,
                self.b_acc_2]

        idxs = T.ivector()
        x = self.index[idxs]
        idxs_r = T.ivector()
        x_r = self.index[idxs_r]
        idxs_s = T.ivector()
        x_s = self.index[idxs_s]
        y_sentence = T.ivector('y_sentence') # labels

        def recurrence_1(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wxi_1) + T.dot(h_tm1,
                self.whi_1) + self.bi_1)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wxf_1) + T.dot(h_tm1,
                self.whf_1) + self.bf_1)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wxo_1) + T.dot(h_tm1,
                self.who_1) + self.bo_1)
            g_t = T.tanh(T.dot(x_t, self.wxg_1) + T.dot(h_tm1, self.whg_1) +
                    self.bg_1)
            c_t = f_t * c_tm1 + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        def recurrence_2(x_t, c_tm1, h_tm1, h_1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wxi_2) + T.dot(h_tm1,
                self.whi_2) + T.dot(h_1, self.wh_1i_2) + self.bi_2)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wxf_2) + T.dot(h_tm1,
                self.whf_2) + T.dot(h_1, self.wh_1f_2) + self.bf_2)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wxo_2) + T.dot(h_tm1,
                self.who_2) + T.dot(h_1, self.wh_1o_2) + self.bo_2)
            g_t = T.tanh(T.dot(x_t, self.wxg_2) + T.dot(h_tm1, self.whg_2) +
                    T.dot(h_1, self.wh_1g_2) + self.bg_2)
            c_t = f_t * c_tm1 + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w_2) + self.b_2)
            return [c_t, h_t, s_t]

        [c_1, h_1], _ = theano.scan(fn=recurrence_1,
                                sequences=x_r,
                                outputs_info=[self.c0_1, self.h0_1],
                                n_steps=x_r.shape[0],
                                truncate_gradient=-1)

        [c_2, h_2, s_2], _ = theano.scan(fn=recurrence_2,
                                sequences=x,
                                non_sequences=[h_1],
                                outputs_info=[T.zeros_like(c_1),
                                    T.zeros_like(h_1), None],
                                n_steps=x.shape[0],
                                truncate_gradient=-1)

        [c_3, h_3, s_3], _ = theano.scan(fn=recurrence_2,
                                sequences=x_s,
                                non_sequences=[h_1],
                                outputs_info=[T.zeros_like(c_1),
                                    T.zeros_like(h_1), None],
                                n_steps=x_s.shape[0],
                                truncate_gradient=-1)

        p_y_given_x_sentence = s_2[:, 0, :]
        p_y_given_x_sentence2 = s_3[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log2(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])

        sentence_gradients = [T.grad(sentence_nll, param) for param in self.params]

        #Adagrad
        sentence_updates = []
        for param_i, grad_i, acc_i in zip(self.params, sentence_gradients, self.params_acc):
            acc = acc_i + T.sqr(grad_i)
            sentence_updates.append((param_i, param_i - lr*grad_i/(T.sqrt(acc)+1e-5)))
            sentence_updates.append((acc_i, acc))

        # SGD
        #sentence_updates = [(param, param - lr*g) for param,g in zip(self.params, sentence_gradients)]

        # theano functions to compile
        #self.classify = theano.function(inputs=[idxs, idxs_r], outputs=y_pred, allow_input_downcast=True)
        #self.prob_dist = theano.function(inputs=[idxs, idxs_r], outputs=p_y_given_x_sentence, allow_input_downcast=True)
        self.prob_dist2 = theano.function(inputs=[idxs_r, idxs_s],
                outputs=p_y_given_x_sentence2, allow_input_downcast=True)
        self.nll = theano.function(inputs=[idxs, idxs_r, y_sentence], outputs=sentence_nll, allow_input_downcast=True)
        self.sentence_train = theano.function(inputs=[idxs, idxs_r, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True)

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder, param.name+'.npy'),
                    param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

def load_data():
    train_file = open('../data/ptb.train.txt', 'r')
    # training set, a list of sentences
    train_set = [l.strip() for l in train_file]
    train_file.close()
    # a list of lists of tokens
    train_set = [l.split() for l in train_set[:100]]
    train_dict = defaultdict(lambda: len(train_dict))
    # an extra symbol for the end of a sentence
    train_dict['<bos>'] = 0
    train_labels = [[train_dict[w] for w in l] for l in train_set]
    train_idxs = [[0]+l[:-1] for l in train_labels]
    train_idxs_r = [l[::-1] for l in train_labels]
    # transform data and label list to numpy array
    train_idxs = [numpy.array(l) for l in train_idxs]
    train_labels = [numpy.array(l) for l in train_labels]

    train_data = (train_idxs, train_idxs_r, train_labels)

    return train_data, train_dict

def ppl(data, rnn):
    nlls = [rnn.nll(x,y,z) for (x,y,z) in zip(data[0], data[1], data[2])]
    mean_nll = numpy.mean(list(nlls))

    return float(2**mean_nll)

def random_generator(probs, vocab_size):
    xk = xrange(vocab_size)
    custm = stats.rv_discrete(name='custm', values=(xk,probs))
    return custm.rvs(size=1)
    return None

def next_word(text, train_dict, index2word, rnn, length):
    words = text.split()
    for j in xrange(20):
        idxs = [train_dict[w] for w in words]
        for i in xrange(length):
            prob_dist = rnn.prob_dist(numpy.asarray(idxs).astype('int32'))
            next_index = random_generator(prob_dist[-1,:], len(train_dict))
            idxs.append(next_index[0])
        print [index2word[index] for index in idxs]

def next_word2(text, text2, train_dict, index2word, rnn, length):
    words_r = text.split()
    words_r = words_r[::-1]
    words_s = text2.split()
    for j in xrange(10):
        idxs_r = [train_dict[w] for w in words_r]
        idxs_s = [train_dict[w] for w in words_s]
        for i in xrange(length):
            prob_dist = rnn.prob_dist2(numpy.asarray(idxs_r).astype('int32'),
                    numpy.asarray(idxs_s).astype('int32'))
            next_index = random_generator(prob_dist[-1,:], len(train_dict))
            #temp = list(prob_dist[-1,:])
            #next_index = temp.index(max(temp))
            idxs_s.append(next_index[0])
        print [index2word[index] for index in idxs_s]

def save_pre_params(folder, pre_params):
    with open(folder+'/pre_params.json', 'wb') as f:
        f.write(json.dumps(pre_params))

def load_pre_params(folder):
    with open(folder+'/pre_params.json', 'rb') as f:
        pre_params = json.loads(f.read())
    return pre_params

def main(param=None):
    if not param:
        param = {
            'lr': 0.1,
            'nhidden': 50,
            # number of hidden units
            'seed': 345,
            'nepochs': 20,
            'savemodel': False,
            'loadmodel': True,
            'folder':'../model/s2s_2',
            'train': False,
            'test': True}
    print param

    # load data and dictionary
    train_data, train_dict = load_data()

    #index2word
    index2word = dict([(v,k) for k,v in train_dict.iteritems()])

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    rnn = RNNLM(nh=param['nhidden'],
                nw=len(train_dict))

    #print rnn.wxg_1.get_value()
    #print rnn.wxg_2.get_value()

    last_ppl = 0
    best_ppl = 10000000000
    lr_current = param['lr']

    # load parameters
    if param['loadmodel'] == True:
        print "loading parameters\n"
        rnn.load(param['folder'])
        (last_ppl, best_ppl, lr_current) = load_pre_params(param['folder'])
        print 'last_ppl: %f' % last_ppl
        print 'best_ppl: %f' % best_ppl
        print 'lr_current: %f' % lr_current

    if param['train'] == True:

        round_num = 20 
        train_data_labels = zip(train_data[0], train_data[1], train_data[2])
        print "Training..."
        start = time.time()
        case_start = time.time()

        for j in xrange(round_num):
            i = 1
            for (x,y,z) in train_data_labels:
                rnn.sentence_train(x, y, z, lr_current)
                if i%50 == 0:
                    case_end = time.time()
                    print "Round %d, case %d, %f seconds" % (j+1, i, case_end-case_start)
                    case_start = time.time()
                i += 1
            print "Testing..."
            test_ppl = ppl(train_data, rnn)
            print "Test perplexity of train data: %f \n" % test_ppl
            last_ppl = test_ppl
            if test_ppl < best_ppl:
                best_ppl = test_ppl
                rnn.save(param['folder'])
            else:
                lr_current /= 2.0
            # save parameters
            print "saving parameters\n"
            pre_params = (last_ppl, best_ppl, lr_current)
            save_pre_params(param['folder'], pre_params)

        end = time.time()
        print "%f seconds in total\n" % (end-start)

        #print rnn.wxg_1.get_value()
        #print rnn.wxg_2.get_value()

    if param['test'] == True:
        text = "there is no asbestos in our products now"
        print 'target: ' + text
        text2 = "<bos>"
        next_word2(text, text2, train_dict, index2word, rnn, 10)
        print '\n'
        text = "we have no useful information on whether users are at risk said james"
        print 'target: ' + text
        text2 = "<bos>"
        next_word2(text, text2, train_dict, index2word, rnn, 10)
        print '\n'
        text = "but you have to recognize that these events took place N years ago"
        print 'target: ' + text
        text2 = "<bos>"
        next_word2(text, text2, train_dict, index2word, rnn, 10)
        print '\n'
        text = "it has no bearing on our work force today"
        print 'target: ' + text
        text2 = "<bos>"
        next_word2(text, text2, train_dict, index2word, rnn, 10)
        print '\n'
        text = "it employs N people and has annual revenue of about $ N million"
        print 'target: ' + text
        text2 = "<bos>"
        next_word2(text, text2, train_dict, index2word, rnn, 10)
        print '\n'


if __name__ == '__main__':
    main()
