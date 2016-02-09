import argparse
import cPickle
import traceback
import logging
import numpy
from theano import config

from imdb import preprocess_data, \
        seq2indices, prepare_reorderdata_minibatch
import time
import sys

logger = logging.getLogger(__name__)
from numpy_compat import argpartition

class BeamSearch(object):

    def __init__(self, enc_dec,beamsize,trgdict,trgdict_rev,srcdict, srcdict_rev):
        self.show_size = 1
        self.enc_dec = enc_dec
        assert 'EOS' in trgdict, 'die'
        assert 'NULL' in trgdict, 'die'
        self.eos_id = trgdict['EOS']
        self.unk_id = trgdict['NULL']
        self.beamsize = beamsize
        self.t_index2word = trgdict_rev
        self.s_index2word = srcdict_rev
        self.s_word2index = srcdict
        self.t_word2index = trgdict
        ff = lambda x: x[:-1][::-1]
        self.reverse_drop_eos_f = lambda x: map(ff,x)
        self.compile()

    def compile(self):
        ## calculate the representation for source sent
        self.comp_repr_enc = self.enc_dec.comp_repr_enc
        self.comp_next_probs_hc = self.enc_dec.decoder.comp_next_probs_hc ## decoding

    def decode(self,src_line):
        seq = src_line.strip().split()
        seq = seq2indices(seq,self.s_word2index)
        if self.enc_dec.reverse_src: seq = seq[::-1]
        tids, costs, best_tran = self.search(seq)
        return costs[0], best_tran

    # return translatgon indices, not include the eos id.
    def kbest_decode(self,src_index):
        seq = src_index
        if self.enc_dec.reverse_src: seq = seq[::-1]
        trans_indice, costs, best_tran = self.search(seq)
        rev_f = lambda x: x[::-1]
        drop_eos_f = lambda x: x[:-1]
        trans_indice = map(drop_eos_f,trans_indice)
        if self.enc_dec.reverse_trg:
            trans_indice = map(rev_f,trans_indice)
        #print 'kbest_decode', zip(trans_indice,costs)
        return trans_indice,best_tran

    def prob_bisent_indice(self,src_indice, trg_indice):
        s_seq = src_indice
        t_seq = trg_indice
        if self.enc_dec.reverse_src: s_seq = s_seq[::-1]
        if self.enc_dec.reverse_trg: t_seq = t_seq[::-1]
        t_seq = t_seq + [self.eos_id]
        #t_seq.append(self.eos_id)
        s_seq = numpy.array(s_seq)[:,None]
        t_seq = numpy.array(t_seq)[:,None]
        s_mask = numpy.ones(s_seq.shape[0], dtype=config.floatX)[:,None]
        t_mask = numpy.ones(t_seq.shape[0], dtype=config.floatX)[:,None]
        return self.enc_dec.f_cost(s_seq,s_mask,t_seq,t_mask)

    def prob_bisent(self,src_line,trg_line):
        s_seq = src_line.strip().split()
        s_seq = seq2indices(s_seq,self.s_word2index)
        if self.enc_dec.reverse_src: s_seq = s_seq[::-1]

        t_seq = trg_line.strip().split()
        if self.enc_dec.reverse_trg: t_seq = t_seq[::-1]
        t_seq.append('EOS')
        t_seq = seq2indices(t_seq,self.t_word2index)

        s_seq = numpy.array(s_seq)[:,None]
        t_seq = numpy.array(t_seq)[:,None]
        s_mask = numpy.ones(s_seq.shape[0], dtype=config.floatX)[:,None]
        t_mask = numpy.ones(t_seq.shape[0], dtype=config.floatX)[:,None]

        return self.enc_dec.f_cost(s_seq,s_mask,t_seq,t_mask)

    def get_sent(self,x,x_mask,i):
        i_len = int(x_mask[:,i].sum())
        return x[:,i][:i_len]


    def minibatch(self,x,x_mask,y,y_mask):
        ### x, y is a minibatch used in training
        sents = numpy.random.permutation(numpy.arange(len(x[0,:])))
        sents = sents[:self.show_size]
        for i in sents:
            seq = self.get_sent(x,x_mask,i)
            seq_t = self.get_sent(y,y_mask,i)
            _,costs,tran = self._search(seq, 1, training=True)
            ref = self.to_words(seq_t[:-1],self.t_index2word)
            if self.enc_dec.reverse_src:
                seq = seq[::-1]
            src = self.to_words(seq,self.s_index2word)
            if self.enc_dec.reverse_trg:
                #tran = " ".join(tran.split()[::-1])
                ref = " ".join(ref.split()[::-1])  ## ref is in minibatch, it is reversed
            print 'score=%f, src: %s\noutput: %s\nref: %s'%(costs[0],src,tran,ref)


    def search(self, seq_origin,  ignore_unk=False, minlen=1, debug=False, training=False):
        assert numpy.array(seq_origin).ndim == 1, 'ndim error'
        n_samples = self.beamsize
        return self._search(seq_origin,n_samples,ignore_unk,minlen,debug,training)

    def _search(self, seq_origin, n_samples, ignore_unk=False, minlen=1, debug=False, training=False):
        ## batch size 1
        '''
        seq = seq_origin
        seqs = numpy.array([seq]*n_samples)
        fin_trans = numpy.array([ [5, 6],[14, 15],])
        #fin_trans = numpy.array([[5, 6]])
        x,x_mask,y,y_mask,_,_ = prepare_reorderdata_minibatch(seqs,fin_trans)
        print x, y
        cost = self.enc_dec.fn_sent_cost(x,x_mask,y,y_mask)
        print 'prob',numpy.log(self.enc_dec.fn_prob(x,x_mask,y,y_mask)+1e-8)
        print 'static cost', cost
        print 'sent static cost', cost.sum(axis=0)
        print 'h_ ', self.enc_dec.fn_proj(x,x_mask,y,y_mask)[:,:,:5]
        print 'x_ ', self.enc_dec.fn_proj_x(x,x_mask,y,y_mask)
        '''
        seq = numpy.array(seq_origin)[:,None]
        mask = numpy.ones(seq.shape,dtype=config.floatX)

        ## to calculate the h and c of encoder, its shape is (d,), where d=4*layers*dim
        h_, c_ = self.comp_repr_enc(seq,mask)## h_.shape=(1,d)
        if debug:
            print "self.enc_dec.layers",self.enc_dec.layers
        ## wrapper hc into shape (1,1,d)
        new_h_ = numpy.tile(h_[0],(1,1,1))
        new_c_ = numpy.tile(c_[0],(1,1,1))

        fin_trans = []
        fin_costs = []
        trans = [[]]#*n_samples## it is the beam, 2d list
        costs = [0.0]
        for k in range(3 * len(seq)):
            if n_samples == 0:
                break
            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))[None,:]
                    if k > 0
                    else numpy.zeros((1,beam_size), dtype="int64"))
            if debug:
                print "last_words, k", last_words, k
            ## given h,c and the last words of trans in beam, to calculate the log_probs with shape(n,v)
            mask = numpy.ones((1,len(last_words[0])),dtype=config.floatX)
            if debug:
                print 'mask',mask
                print "new_h_.shape",new_h_.shape
                print "new_c_.shape",new_h_.shape
                print "last_words, k", last_words, k
                print self.enc_dec.decoder.dbg(last_words,mask,new_h_[0],new_c_[0], 0.0)
            log_probs, h_, c_, proj_x = self.comp_next_probs_hc(last_words,mask,new_h_[0],new_c_[0], 1.0) \
                if k>0 else self.comp_next_probs_hc(last_words,mask,new_h_[0],new_c_[0], 0.0)
            log_probs = numpy.log(log_probs[0])

            if debug:
                print 'new_h_[0,:,:5]',new_h_[0,:,:5]
                print 'h_[0,:,:5]',h_[0,:,:5]
                print 'proj_x', proj_x
                print 'log_probs', log_probs

            if k > 0 and 0:
                last_words[0][-1]=5
                print last_words
                log_probs_new, _, _, _ = self.comp_next_probs_hc(last_words,mask,new_h_,new_c_, 1.0)
                log_probs_new = numpy.log(log_probs_new[0])
                print 'log_probs', log_probs_new

            if debug:
                print 'shape log_probs', log_probs.shape ## its shape is (n,v), v is the vocab size
            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs

            if debug:
                print "next_costs.shape",next_costs.shape
            #print next_costs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]

            ## trans_indices is indicate the previous trans id.
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            if debug:
                print 'best_costs_indices',best_costs_indices
                print 'trans_indices',trans_indices
                print 'word_indices',word_indices
            costs = flat_next_costs[best_costs_indices]
            #print costs
            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                inputs[i] = next_word

            #print 'new_costs', new_costs
            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.eos_id:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
            if debug:
                print 'new_trans',new_trans
                print 'trans',trans

                print 'h_.shape',h_.shape
                print 'indices',indices
                print 'trans_indices',trans_indices

            pre_t_indices = trans_indices[indices]
            wrapper_fn = lambda x: (x[pre_t_indices])[None,:]
            #print 'pre_t_indices',pre_t_indices
            #print 'h_[0,:,:5]',h_[0,:,:5]
            new_h_ = wrapper_fn(h_[0])
            new_c_ = wrapper_fn(c_[0])

            if debug:
                print 'new_h_.shape',new_h_.shape
            if k == 1:
                pass #break

        if debug:
            print 'fin_trans',fin_trans
        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 4 and not training:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                print 'seq is empty?', seq_origin
                return self._search(seq_origin, n_samples * 2, False, minlen)
            else:
                logger.warning("Translation failed: cannot end with EOS")
                return [[]],[0.0],"NO TRANS"
        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        #'''
#        print 'fin_trans',fin_trans
        #print 'fin_costs',fin_costs
        seq = seq_origin
        seqs = numpy.array([seq]*len(fin_trans))
        #print seqs.shape,seqs
        x,x_mask,y,y_mask,_,_ = prepare_reorderdata_minibatch(seqs,fin_trans)
        #print "x", x, x_mask
        #print "y", y, y_mask
        cost = self.enc_dec.fn_sent_cost(x,x_mask,y,y_mask)
        #print 'static cost', cost
        #print 'sent static cost', cost.sum(axis=0)
        #'''
        #print 'trans id', fin_trans[0][:-1]
        #print "dict", self.t_index2word
        best_trans = fin_trans[0][:-1]
        if self.enc_dec.reverse_trg: best_trans = best_trans[::-1]
        best_trans = self.to_words(best_trans,self.t_index2word)
        return list(fin_trans), fin_costs, best_trans

    def to_words(self,seq,index2word):
        return " ".join(seq2indices(seq,index2word))
