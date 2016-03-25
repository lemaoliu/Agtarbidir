import cPickle
import gzip
import os
import sys
from collections import defaultdict

import numpy
import theano


def pkl2txt(path):

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    f.close()

    def save(xlist, filename):
        ff = open(filename,'w')
        for x in xlist: 
            print >>ff, x
        ff.close()

    save(train_set[0],"train_x.txt")
    save(train_set[1],"train_y.txt")

       
def get_label(align,distortion_limit=10):

    ## relative distortion: -dl, -dl+1, ..., -1, 0, 1, ..., dl, NULL
    ## its corresponding class: 0,      1,  ..., -1+dl, dl, dl+1, ..., 2*dl, 2*dl+1
    num_classes = 2*distortion_limit + 2
    label = [0]*len(align)
    pre_null_p, cur_p = -1, 0
    while cur_p < len(align):
        if align[cur_p] != -1:
            if cur_p == 0:
                ## translate the first src word
                label[cur_p] = align[cur_p] + distortion_limit + 1 #if align[cur_p] >0 \
                    #else distortion_limit + 1 
            elif pre_null_p != -1:
                label[cur_p] = align[cur_p] - align[pre_null_p] + distortion_limit
            else: # cur_p>0 and pre_null_p == -1
                ## translate the first src word after inserting several words previously
                label[cur_p] = align[cur_p] + distortion_limit + 1#if align[cur_p] >0 \
                    #else distortion_limit + 1          
            pre_null_p = cur_p

            label[cur_p] = min(2*distortion_limit,label[cur_p])# or label[cur_p] < 0:
            label[cur_p] = max(0,label[cur_p])# or label[cur_p] < 0:
        else:
            label[cur_p] = num_classes-1        
        cur_p +=1        
    '''
    print align
    print label
    print zip(align,label)
    '''
    return label

def prepare_reorderdata_minibatch(seqs_x, seqs_y, seqs_align=None, seqs_label=None, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    def padding(seqs,maxlen):
        lengths = [len(s) for s in seqs]
    
        if maxlen is not None:
            new_seqs = []
            new_lengths = []
            for l, s in zip(lengths, seqs):
                if l < maxlen:
                    new_seqs.append(s)
                    new_lengths.append(l)
            lengths = new_lengths
            seqs = new_seqs
    
            if len(lengths) < 1:
                return None, None, None
    
        n_samples = len(seqs)
        maxlen = numpy.max(lengths)
    
        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.
        return x, x_mask
    x, x_mask = padding(seqs_x,maxlen)
    y, y_mask = padding(seqs_y,maxlen)
    aligns, labels = None, None
    if seqs_align is not None:
        aligns, _ = padding(seqs_align,maxlen)
    if seqs_label is not None: 
        labels, _ = padding(seqs_label,maxlen)

    return x, x_mask, y, y_mask, aligns, labels

def split_train(train_set,maxlen=None,
              valid_test_portion=0.2,
              sort_by_len=True):

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_a = []
        new_train_set_l = []
        for x, y, a,l in zip(train_set[0], train_set[1],train_set[2],train_set[3]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_a.append(a)
                new_train_set_l.append(l)
        train_set = (new_train_set_x, new_train_set_y,new_train_set_a,new_train_set_l)
        del new_train_set_x, new_train_set_y, new_train_set_a, new_train_set_l

    # split training set into validation set
    train_set_x, train_set_y, train_set_a, train_set_l = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_test_portion)))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_a = [train_set_a[s] for s in sidx[n_train:]]
    valid_set_l = [train_set_l[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_a = [train_set_a[s] for s in sidx[:n_train]]
    train_set_l = [train_set_l[s] for s in sidx[:n_train]]

    n_valid = int(numpy.round(len(valid_set_x) * 0.5))
    test_set_x = valid_set_x[:n_valid]
    test_set_y = valid_set_y[:n_valid]
    test_set_a = valid_set_a[:n_valid]
    test_set_l = valid_set_l[:n_valid]

    valid_set_x = valid_set_x[n_valid:]
    valid_set_y = valid_set_y[n_valid:]
    valid_set_a = valid_set_a[n_valid:]
    valid_set_l = valid_set_l[n_valid:]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        test_set_a = [test_set_a[i] for i in sorted_index]
        test_set_l = [test_set_l[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_a = [valid_set_a[i] for i in sorted_index]
        valid_set_l = [valid_set_l[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_a = [train_set_a[i] for i in sorted_index]
        train_set_l = [train_set_l[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_a, train_set_l)
    valid = (valid_set_x, valid_set_y, valid_set_a, valid_set_l)
    test = (test_set_x, test_set_y, test_set_a, test_set_l)

    return train, valid, test

def load_dict(dictfilename, bidirection=False):

    npdict = cPickle.load(open(dictfilename,'rb'))
    word_dict = defaultdict(int)
    for k,v in npdict.iteritems():
        word_dict[str(k)] = int(v)
    if not bidirection:
        return word_dict
    else:
        dict_rev = defaultdict()
        for k,v in npdict.iteritems():
            dict_rev[int(v)] = str(k) #'file' if str(k) == '$filefile$' else str(k)
        return word_dict, dict_rev

def seq2indices(words,word_dict):
    line_ids = []
    for w in words:
        #if w == 'file': w = '$filefile$'### required by numpy.savez
        line_ids.append(word_dict[w])      
    return line_ids


def preprocess_data(src,trg,align=None,srcdict=None,trgdict=None,sort_by_len=True):
    from collections import defaultdict
    def load(filename,dict=None):
        word_dict = defaultdict(int) if dict is None else load_dict(dict) 
        tmpid = 2 ## if it is not 2, for example 3, it may crash when look up the key 3.
        train_set=[]
        word_dict['NULL'] = 0 
        word_dict['EOS'] = 1 
        for line in open(filename,'r'):
            words = line.strip().split()
            line_ids = []
            for w in words:
                #if w == 'file': w = '$filefile$'### required by numpy.savez
                if word_dict[w] == 0:
                    if dict is None:### construct the word_dict
                        word_dict[w] = tmpid
                        tmpid += 1
                    else:
                        word_dict[w] = word_dict['NULL']
            
                line_ids.append(word_dict[w])      
            train_set.append(line_ids)
        return train_set,word_dict

           
    src_set, src_dict = load(src,srcdict) 
    trg_set, trg_dict = load(trg,trgdict)


    trg_dict['EOS'] = 1
    for x in trg_set:## add eos to each line.
        x.append(1)

    if srcdict is None:
        cPickle.dump(src_dict,open('src.dict.pkl','wb')) 
        cPickle.dump(trg_dict,open('trg.dict.pkl','wb')) 

    def modify_align(align,trg_set):
        aligns = open(align,'r').readlines()
        align_set, label_set = [], []
        for i, line in enumerate(aligns):
            trg_len = len(trg_set[i])
            align_line = []
            align_dict = {}
            for a in line.strip().split():
                s2t = tuple(map(int,a.split('-')))
                if s2t[1] not in align_dict:
                    align_dict[s2t[1]] = s2t[0]
                else:
                    from numpy import random
                    if random.rand() > 0.5:
                        align_dict[s2t[1]] = s2t[0]

            for ii in xrange(trg_len):
                if ii not in align_dict:
                    align_dict[ii] = -1
                align_line.append(align_dict[ii])
            align_set.append(align_line)
            label_set.append(get_label(align_line))

        return align_set, label_set
    align_set, label_set = modify_align(align,trg_set) if align is not None else None, None


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(src_set)
        src_set = [src_set[i] for i in sorted_index]
        trg_set = [trg_set[i] for i in sorted_index]
        if  align_set is not None:
            align_set = [align_set[i] for i in sorted_index]
            label_set = [label_set[i] for i in sorted_index]

    return (src_set, trg_set, align_set, label_set), \
            numpy.max(src_dict.values())+1, numpy.max(trg_dict.values())+1


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


