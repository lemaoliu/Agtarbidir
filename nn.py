from collections import OrderedDict
import cPickle as pkl
import sys
import time
import logging
import re

import heapq
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from imdb import preprocess_data, prepare_reorderdata_minibatch, \
                 get_label, load_dict, seq2indices
from imdb import split_train
import imdb
import os
from theano.ifelse import ifelse

from search import BeamSearch

import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu3')

logger = logging.getLogger(__name__)
SEED = 123
numpy.random.seed(SEED)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def softmax(x):

    if x.ndim == 2:
        e = tensor.exp(x)
        return e / tensor.sum(e, axis=1).dimshuffle(0, 'x')
    elif x.ndim == 3:
        e = tensor.exp(x)
        return e / tensor.sum(e, axis=2).dimshuffle(0, 1,'x')
    elif x.ndim == 1:
        e = tensor.exp(x)
        return e/ tensor.sum(e)
    else:
        die

def numpy_int(data):
    return numpy.asarray(data, dtype='int64')

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    else:
        return _x[:, n * dim:(n + 1) * dim]

def _p(prefix,pre):
        return "%s_%s"%(prefix,pre)


def print_tparams(tparams):
    print 'the sum'
    for v in tparams:
        print v.name,v.get_value().sum(),
    print ''



class Trainer(object):

    def __init__(self, coeff=0.95, diag=1e-6, eta=2):
        self.coeff = coeff
        self.diag = diag
        self.eta = eta
        self.use_sgd = 0

    def sgd(self, lr, inputs, cost, grads, tparams):
        """ Stochastic Gradient Descent

        :note: A more complicated version of sgd then needed.  This is
            done like that for adadelta and rmsprop.

        """
        # New set of shared variable that will contain the gradient
        # for a mini-batch.
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name)
                   for p in tparams]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        # Function that computes gradients for a mini-batch, but do not
        # updates the weights.
        f_grad_shared = theano.function(inputs, cost, updates=gsup,
                                        name='sgd_f_grad_shared')

        pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]

        # Function that updates the weights from the previously computed
        # gradient.
        f_update = theano.function([lr], [], updates=pup,
                                   name='sgd_f_update')

        return f_grad_shared, f_update


    def SetupTainer(self, lr, inputs, cost, tparams):

        grads = tensor.grad(cost, wrt=tparams, disconnected_inputs='ignore')
        #print "grads",type(grads), type(grads[0])
        #print "grads",map(lambda x: x.name, grads)
        #print "tparams",tparams
        grads = clip_grad(grads,tparams)

        if self.use_sgd:
            return self.sgd(lr, inputs, cost, grads, tparams)
        else:
            return self.adadelta(lr, inputs, cost, grads, tparams)

    def adadelta(self, lr, inputs, cost, grads, tparams):
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % p.name)
                        for p in tparams]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % p.name) ## delte_x_square_sum
                       for p in tparams]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % p.name)
                          for p in tparams] ### g_square_sum

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]

        rg2up = [(rg2, self.coeff * rg2 + (1-self.coeff) * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
        ### update the g_square_sum
        f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared',on_unused_input='ignore')

        updir = [-tensor.sqrt(ru2 + self.diag) / tensor.sqrt(rg2 + self.diag) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, self.coeff * ru2 + (1-self.coeff) * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + self.eta*ud) for p, ud in zip(tparams, updir)]

        ## update the delte_x_square_sum and parameters
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')

        return f_grad_shared, f_update


def clip_grad(grads,params, max_norm=numpy_floatX(5.)):

    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    not_finite = tensor.or_(tensor.isnan(grad_norm), tensor.isinf(grad_norm))
    grad_norm = tensor.sqrt(grad_norm)
    cgrads = []
    for g,p in zip(grads,params):
        tmpg = tensor.switch(tensor.ge(grad_norm, max_norm), g*max_norm/grad_norm, g)
        ttmpg = tensor.switch(not_finite, numpy_floatX(0.1)*p,tmpg)
        cgrads.append(ttmpg)

    return cgrads


class Layer(object):

    def __init__(self,dim,prefix=""):
        self.dim = dim
        self.prefix = self.prefix

    def pp(self,pre):
        return _p(self.prefix,pre)

    def ortho_weight(self,ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)

    def ortho_weight_copies(self, d, n):
        Ws = [self.ortho_weight(d) for i in xrange(n)]
        return numpy.concatenate(Ws,axis=1)

    def rand_weight(self,shape):## shape is a tuple
        randn = numpy.random.rand(*shape)
        return (0.01 * randn).astype(config.floatX)

class Layer_emb(Layer):

    def __init__(self,dim,vcbsize,prefix=""):
        self.prefix = prefix
        self.dim = dim
        self.vcbsize = vcbsize
        self.emb_W = theano.shared(self.rand_weight((vcbsize,dim)),name=self.pp('emb'))
        self.params = [self.emb_W]

    ## x.shape=(timesteps,n)
    ## x is a list (sentence) or a list of list (minibatch of sentences)

    def fprop(self,x):
        #print "emb dim", self.dim
        emb = self.emb_W[x.flatten()]
        #print "x.ndim",x.ndim
        if x.ndim == 1:
            return emb
        else:
            return tensor.reshape(emb,(x.shape[0],x.shape[1],self.dim))
            #return emb.reshape((x.shape[0],x.shape[1],self.dim))

class Lstm_layer(Layer):

    def __init__(self,dim,layers,prefix=""):
        #super(Layer, self).__init__(dim,prefix)
        self.dim = dim
        self.prefix = prefix
        self.layers = layers## it is not tensor variable
        self.W_hi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers), name=self.pp('W_hi')) ### (d, layers*4*d), reuse the history hiden as input of lstm net, each _slice is used for one layer
        self.W_xi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers),name=self.pp('W_xi')) ### (d, layers*4*d), reuse the input x as input of lstm net
        self.W_lhi = theano.shared(self.ortho_weight_copies(self.dim, 4*self.layers), name=self.pp('W_lhi')) ### (d, layers*4*d), reuse the lower hiden layer as input of lstm net
        self.bx = theano.shared(numpy.zeros((4*self.layers*self.dim,)).astype(config.floatX),name=self.pp('W_bx')) ### (1,layers*4*d)

        self.params = [self.W_hi,self.W_xi,self.W_lhi,self.bx]

    ### h_.shape=(n,layers*d), where n is the batch size
    ### x_.shape=(n,layers*4*d)
    ### c_.shape=(n,layers*d)
    def _step(self, m_, x_, h_, c_):
        n_samples = h_.shape[0]
        assert h_.ndim == 2, 'ndim error in _step func'
        hs, cs = [], []
        zeros = tensor.alloc(numpy_floatX(0.), n_samples, self.dim)

        for l in xrange(self.layers):
            ##  _slice(self.W_hi,l,4*self.dim), reuse the history as input, (d, 4*d)
            w_h = tensor.dot(_slice(h_,l,self.dim), _slice(self.W_hi,l,4*self.dim))
            ht_ = w_h + _slice(x_,l,4*self.dim)  ## shape=(n,4*d)
            hl_ = hs[-1] if len(hs) > 0 else zeros

            ## reuse the lower hiden layer as input of lstm net
            ht_ += tensor.dot(hl_,_slice(self.W_lhi,l,4*self.dim))
            ct_ = _slice(c_,l,self.dim)

            i = tensor.nnet.sigmoid(_slice(ht_, 0, self.dim))
            f = tensor.nnet.sigmoid(_slice(ht_, 1, self.dim))
            o = tensor.nnet.sigmoid(_slice(ht_, 2, self.dim))
            c = tensor.nnet.sigmoid(_slice(ht_, 3, self.dim))

            c = f * ct_ + i * c
            h = o * tensor.tanh(c)

            hs.append(h)
            cs.append(c)

        h = tensor.concatenate(hs,axis=1) ### (n,layers*d)
        c = tensor.concatenate(cs,axis=1) ### (n,layers*d)
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    ### state_below is the lower representation,
    def fprop(self,state_below,mask=None,start_h=None,start_c=None):

        n_timesteps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if start_h is None:
            start_h = tensor.alloc(numpy_floatX(0.), n_samples, self.layers*self.dim)
        if start_c is None:
            start_c = tensor.alloc(numpy_floatX(0.), n_samples, self.layers*self.dim)

        ## shape=(n,layers*4*d)
        state_below = tensor.dot(state_below,self.W_xi) + self.bx
        rval, _ = theano.scan(self._step,
                                    sequences=[mask, state_below],
                                    outputs_info=[start_h,start_c],
                                    name=_p(self.prefix, '_layers'),
                                    n_steps=n_timesteps)
        return rval[0], rval[1] ### return h and c for all timesteps and all layers

    def BuildEnc(self,state_below,mask):
        all_h,all_c = self.fprop(state_below,mask)
        return all_h[-1], all_c[-1]

    '''
    def get_repr(self,state_below,mask=None):
        state_below = self.get_repr(state_below,mask)
        return state_below[-1]## shape=(n,d)
        return rval[0][:,:,beg:] #_slice(rval[0][-1],self.layers-1,self.dim) ### (t,n,d)
    '''



class Lstm_decoder(Lstm_layer):

    def __init__(self,dim,emb,layers,prefix=""):
        self.sum_over_time = True
        #super(Lstm_layer, self).__init__(dim,layers,prefix)
        Lstm_layer.__init__(self,dim,layers,prefix)
        self.emb = emb
        self.U = theano.shared(self.rand_weight((self.dim,self.emb.vcbsize)),name=_p(self.prefix,'U')) ### (d,self.emb.vcbsize)
        self.b = theano.shared(numpy.zeros((self.emb.vcbsize,)).astype(config.floatX), name=self.pp('b')) ### (1,self.emb.vcbsize)
        self.params += [self.U,self.b]


    ''' for decoding, given h_,c_, y, and y_mask'''
    def Step_decoding(self):
        ## if i_y = None, then it means decoding the first word

        i_y = tensor.matrix('i_y', dtype='int64')
        i_y_mask = tensor.matrix('i_y_mask', dtype=config.floatX)
        h_ = tensor.matrix('h', dtype=config.floatX)
        c_ = tensor.matrix('c', dtype=config.floatX)
        flag = tensor.scalar('flag',dtype=config.floatX)
        state_below = tensor.alloc(numpy_floatX(0.), i_y_mask.shape[0], i_y_mask.shape[1], self.dim)
        ## shape=(1,n,d)
        shape = (i_y.shape[0],i_y.shape[1],self.dim)
        #i_y_repr = self.emb.emb_W[i_y.flatten()].reshape(shape)
        i_y_repr = self.emb.fprop(i_y)


        state_below = ifelse(tensor.gt(flag, 0.5), i_y_repr, state_below)
        #state_below = tensor.switch(tensor.gt(flag, 0.5), self.emb.fprop(i_y), state_below)
        proj_h, proj_c = self.fprop(state_below,i_y_mask,h_,c_)
        proj_h, proj_c = proj_h[0],proj_c[0]

        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        proj_xx = tensor.dot(final_layer_h, self.U)
        proj_x = proj_xx + self.b
        assert proj_h.ndim == 2, 'ndim error'
        self.dbg = theano.function([i_y,i_y_mask,h_,c_,flag],\
                    [proj_h, self.U.shape, self.b.shape],on_unused_input='ignore')
        prob = softmax(proj_x)
        self.comp_next_probs_hc = theano.function([i_y,i_y_mask,h_,c_,flag], \
            [tensor.shape_padleft(prob), tensor.shape_padleft(proj_h), tensor.shape_padleft(proj_c), proj_x, ])



    ''' for decoding, given h_,c_, y, and y_mask'''
    def Step_decoding1(self):
        ## if i_y = None, then it means decoding the first word

        i_y = tensor.matrix('i_y', dtype='int64')
        i_y_mask = tensor.matrix('i_y_mask', dtype=config.floatX)
        h_ = tensor.tensor3('h', dtype=config.floatX)
        c_ = tensor.tensor3('c', dtype=config.floatX)
        flag = tensor.scalar('flag',dtype=config.floatX)
        state_below = tensor.alloc(numpy_floatX(0.), i_y_mask.shape[0], i_y_mask.shape[1], self.dim)
        ## shape=(1,n,d)
        shape = (i_y.shape[0],i_y.shape[1],self.dim)
        #i_y_repr = self.emb.emb_W[i_y.flatten()].reshape(shape)
        i_y_repr = self.emb.fprop(i_y)


        state_below = ifelse(tensor.gt(flag, 0.5), i_y_repr, state_below)
        #state_below = tensor.switch(tensor.gt(flag, 0.5), self.emb.fprop(i_y), state_below)
        proj_h, proj_c = self.fprop(state_below,i_y_mask,h_,c_)
        proj_h, proj_c = proj_h[0],proj_c[0]
        print 'proj_h.ndim', proj_h.ndim
        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        proj_xx = tensor.dot(final_layer_h, self.U)
        proj_x = proj_xx + self.b
        assert proj_h.ndim == 3, 'ndim error'
        self.dbg = theano.function([i_y,i_y_mask,h_,c_,flag],\
                    [proj_h, self.U.shape, self.b.shape],on_unused_input='ignore')
        prob = softmax(proj_x)
        self.comp_next_probs_hc = theano.function([i_y,i_y_mask,h_,c_,flag], \
            [prob, proj_h, proj_c, proj_x, ])

    ''' compute the cost (-log likelyhood) for training'''
    def Cost(self, y, y_mask, h_=None, c_=None):

        enc_y = self.emb.fprop(y)## shape=(t,n,d)
        #h_ = _slice(h_c,0,4*self.dim) if h_c is not None else None
        #c_ = _slice(h_c,1,4*self.dim) if h_c is not None else None
        n_samples = y.shape[1]
        beg_y = tensor.alloc(numpy_floatX(0.), n_samples, self.dim)
        state_below = tensor.concatenate((tensor.shape_padleft(beg_y),enc_y[:-1]),axis=0)
        proj_h, proj_c = self.fprop(state_below,y_mask,h_,c_)

        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        self.proj_h = proj_h
        self.proj_c = proj_c
        self.final_layer_h = final_layer_h

        proj_x = tensor.dot(final_layer_h, self.U) + self.b
        assert proj_x.ndim == 3, 'ndim error'
        self.proj_x = proj_x
        pred = softmax(proj_x)
        self.pred = pred
        #self.one_step_decoder = theano.function([y,y_mask,h_,c_],[pred, self.proj_h,self.proj_c])
        self.argpred = pred.argmax(axis=2)
        fr = tensor.reshape(pred,(pred.shape[0]*pred.shape[1],pred.shape[2]))
        lr = tensor.reshape(y,(y.shape[0]*y.shape[1],))
        B = tensor.arange(fr.shape[0])
        l_pred = fr[B,lr[B]] ## it is a vector
        #l_pred = tensor.reshape(label.shape[0],label.shape[1])
        #print "y_mask.ndim",y_mask.ndim
        y_mask_r = tensor.reshape(y_mask,(y_mask.shape[0]*y_mask.shape[1],))
        assert l_pred.ndim == 1, "dimmension error"
        ### error
        cost = -tensor.log(l_pred + 1e-8)
        cost = cost * y_mask_r
        cost_matrix = tensor.reshape(cost,(y_mask.shape[0],y_mask.shape[1]))
        sent_cost = cost_matrix.sum(axis=0)
        cost = cost.sum()
        if self.sum_over_time:
            cost = cost /tensor.cast(y_mask.shape[1],dtype=config.floatX)
        else:
            cost = cost/y_mask_r.sum()
        return cost, sent_cost

    ''' compute the cost (-log likelyhood) for training'''
    def Cost_local(self, y, y_mask, vcb, l_y, v_mask, h_=None, c_=None):
        ### vcb=imatrix (t,n,v), l_y (t,n,v) is the position index of y in vcb
        enc_y = self.emb.fprop(y)## shape=(n,d)
        #h_ = _slice(h_c,0,4*self.dim) if h_c is not None else None
        #c_ = _slice(h_c,1,4*self.dim) if h_c is not None else None
        n_samples = y.shape[1]
        beg_y = tensor.alloc(numpy_floatX(0.), n_samples, self.dim)
        state_below = tensor.concatenate((tensor.shape_padleft(beg_y),enc_y[:-1]),axis=0)
        proj_h = self.fprop(state_below,y_mask,h_,c_)[0]

        final_layer_h = _slice(proj_h,self.layers-1,self.dim)
        self.proj_h = proj_h
        self.final_layer_h = final_layer_h
        ###calculate the local cost (i.e. normalized wrt vcb) for each sentence at each timestep
        ### h.shape=(d,), v.shape=(v,)

        def _lcl_cost(h, v, l_y, v_mask):
            end = tensor.cast(v_mask.sum(),dtype=v.dtype)
            vv = v[:end]
            U = self.U[:,vv]
            b = self.b[:,vv]
            proj_x = tensor.dot(h,U) + b
            pred = softmax(proj_x)
            return -tensor.log(pred[l_y] + 1e-8)

        def Reshape3d(x):
            return tensor.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))

        h_r = Reshape3d(final_layer_h)
        v_r = Reshape3d(vcb)
        l_y_r = Reshape3d(l_y)
        y_mask_r = Reshape3d(y_mask)
        v_mask_r = Reshape3d(v_mask)

        ravls, _ = theano.scan(_lcl_cost,
                                    sequences=[h_r,v_r, l_y_r, v_mask_r],
                                    outputs_info=[None],
                                    name=_p(self.prefix, '_cost'),
                                    n_steps=h_r.shape[0])
        cost = ravls * y_mask_r
        cost = cost.sum()/y_mask_r.sum()
        return cost


def get_options(mfile):
    opts = numpy.load(mfile)
    conf_opt = OrderedDict()
    conf_opt['dim_proj'] = int(opts['dim_proj'])
    conf_opt['layers'] = int(opts['layers'])
    conf_opt['n_words_x'] = int(opts['n_words_x'])
    conf_opt['n_words_y'] = int(opts['n_words_y'])
    conf_opt['reverse_src'] = bool(opts['reverse_src'])
    del opts
    return conf_opt


class EncoderDecoder(object):

    def __init__(self, config, emb_x=None, emb_y=None, prefix=''):
        ### basic config
        self.prefix = prefix
        self.config = config
        self.dim = config['dim_proj']
        self.layers = config['layers']
        self.srcvcbsize = config['n_words_x']
        self.trgvcbsize = config['n_words_y']
        self.reverse_src = config['reverse_src']
        self.reverse_trg = config['reverse_trg']

        self.emb_x = emb_x if emb_x is not None else \
                     Layer_emb(self.dim,self.srcvcbsize,'%s_src'%prefix)
        self.emb_y = emb_y if emb_y is not None else \
                     Layer_emb(self.dim,self.trgvcbsize,'%s_trg'%prefix)
        self.encoder = Lstm_layer(self.dim,self.layers,'%s_enc'%prefix)
        self.decoder = Lstm_decoder(self.dim,self.emb_y,self.layers,'%s_dec'%prefix)
        self.cost = None
        ### model params for training
        if emb_x is None:
            self.params = self.emb_x.params + self.emb_y.params + \
                          self.encoder.params + self.decoder.params
        else:
            self.params = self.encoder.params + self.decoder.params

    def unzip(self):
        new_params = OrderedDict()
        for p in self.params:
            new_params[p.name] = p.get_value()
        return new_params

    def add_dict(self, ori_dict, adder_dict):
        for kk, vv in adder_dict.iteritems():
            ori_dict[kk] = vv

    def save(self,mfile,params=None, **info):
        new_params = self.unzip() if params is None else params
        self.add_dict(new_params,info)
        self.add_dict(new_params,self.config)
        numpy.savez(mfile,**new_params)

    def zipp(self,new_params):
        for p in self.params:
            p.set_value(new_params[p.name])


    def load(self,mfile):
        new_params = numpy.load(mfile)
        self.zipp(new_params)
        '''
        for p in self.params:
            p.set_value(new_params[p.name])
        '''

    def BuildEncDec(self):
        use_noise = theano.shared(numpy_floatX(0.))
        x = tensor.matrix('x', dtype='int64')
        y = tensor.matrix('y', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype=config.floatX)
        y_mask = tensor.matrix('y_mask', dtype=config.floatX)
        self.inputs = [x,x_mask,y,y_mask]
        logger.debug("To Build compuatation graph")
        enc_x = self.emb_x.fprop(x)
        self.x_repr_fn = theano.function([x],enc_x)
        #self.fn_enc = theano.function(self.inputs, enc_x,on_unused_input='ignore')
        logger.debug("To Build encoding graph")
        ## the last timestep and all layers for h and c
        h_, c_ = self.encoder.BuildEnc(enc_x,x_mask)
        self.h_, self.c_ = h_, c_
        self.fn_enc = theano.function(self.inputs,c_,on_unused_input='ignore')
        logger.debug("Building encoding graph over")
        logger.debug("To Build decoding graph")
        self.cost, self.sent_cost = self.decoder.Cost(y,y_mask,h_,c_)
        self.fn_proj = theano.function(self.inputs,self.decoder.proj_h,on_unused_input='ignore')
        self.fn_proj_x = theano.function(self.inputs,self.decoder.proj_x,on_unused_input='ignore')
        self.fn_prob = theano.function(self.inputs,self.decoder.pred,on_unused_input='ignore')
        self.fn_final = theano.function(self.inputs,self.decoder.final_layer_h,\
                            on_unused_input='ignore')

        self.fn_sent_cost = theano.function(self.inputs,self.sent_cost)
        self.comp_repr_enc = theano.function([x,x_mask],[self.h_,self.c_])
        logger.debug("Building decoding graph over")
        logger.debug("To Build timestep wise decoder")
        self.cost, self.sent_cost = self.decoder.Cost(y,y_mask,h_,c_)
        self.decoder.Step_decoding()
        logger.debug("Building timestep wise decoder over")
        self.f_cost = theano.function(self.inputs, self.cost, name='f_cost',\
            on_unused_input='ignore')
        self.f_sent_cost = theano.function(self.inputs, self.sent_cost, \
            name='f_sent_cost',on_unused_input='ignore')
        return self.inputs, self.cost, use_noise


def pred_error(f_pred, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        y = [data[1][t] for t in valid_index]
        align = [data[2][t] for t in valid_index]
        label = [data[3][t] for t in valid_index]
        x, x_mask, y, y_mask, align, label = \
            prepare_reorderdata_minibatch(x, y, align, label)
        preds = f_pred(x, x_mask, y, y_mask)
        targets = numpy.array(y)
        valid_err += ((preds == targets)*y_mask).sum()/y_mask.sum()
        if verbose:
            print "---- batch ----"
            print "predictions == labels?"
            print preds == targets
            print "preds", preds
            print "targets", targets
            print "mask",y_mask
    valid_err = 1. - numpy_floatX(valid_err) / len(iterator)
    return valid_err

def train_lstm(
    dim_proj=10,  # word embeding dimension and LSTM number of hidden units.
    layers=2, # the number of layers for lstm encoder and decoder
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    begin_valid=0,## when begin to evalute the performance on dev and test sets.
    lrate=10,  # Learning rate for sgd (not used for adadelta and rmsprop)
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model',  # The best model will be saved there
    save_on_the_fly=0,
    validFreq=-1,#370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=3,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',
    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,#True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    src_file="3.zh",
    trg_file="3.en",
    align_file="3.align",
    src_dict='src.dict.pkl',
    trg_dict='trg.dict.pkl',
    dev_file='',
    dev_ref='',
    dev_xml='',
    tst_file='',
    tst_ref='',
    tst_xml='',
    reverse_src=True,
    reverse_trg=False,
    bi_train=False,
    bi_reg=1.0,
    beamsize=12,
):
    # Model options
    model_options = locals().copy()
    print 'Loading data'
    train, n_words_x, n_words_y = preprocess_data(src_file,trg_file)

    srcdict, srcdict_rev = load_dict(src_dict,True)
    trgdict, trgdict_rev = load_dict(trg_dict,True)


    print 'the tot number of examples is %d '%len(train[0])
    if reverse_src:
        for i in xrange(len(train[0])):
            train[0][i] = train[0][i][::-1]

    numpy.savez('train_data.npz',train)
    print "n_words_x",n_words_x
    print "n_words_y",n_words_y

    #_, valid, test = split_train(train)
    valid, test = train, train
    model_options['n_words_x'] = n_words_x
    model_options['n_words_y'] = n_words_y

    print "model options", model_options
    logger.debug("Building model")

    ## generate the target from left to right
    enc_dec = EncoderDecoder(model_options,prefix='l2r')

    if reload_model:
        enc_dec.load(saveto+'_f.npz')

    inputs, cost, use_noise = enc_dec.BuildEncDec()

    rev_model_options = locals().copy()
    rev_model_options['n_words_x'] = n_words_x
    rev_model_options['n_words_y'] = n_words_y
    rev_model_options['reverse_src'] = False
    rev_model_options['reverse_trg'] = True

    print 'building the right to left model'
    ## generate the target from right to left, and it doesnot share the params with enc_dec
    rev_enc_dec = EncoderDecoder(rev_model_options,prefix='r2l')
    if reload_model:
        rev_enc_dec.load(saveto+'_r.npz')

    rev_inputs, rev_cost, rev_use_noise = rev_enc_dec.BuildEncDec()

    inputs = inputs + rev_inputs
    cost = bi_reg*cost + rev_cost
    params = enc_dec.params + rev_enc_dec.params

    f_bi_cost = theano.function(inputs,cost,on_unused_input='ignore')
    print 'total params: ', params

    beamsearch =     BeamSearch(    enc_dec, beamsize, trgdict, trgdict_rev,srcdict, srcdict_rev)
    beamsearch_rev = BeamSearch(rev_enc_dec, beamsize, trgdict, trgdict_rev,srcdict, srcdict_rev)
    print 'the initialized params'
    print_tparams(beamsearch.enc_dec.params)
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (enc_dec.decoder.U ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = enc_dec.f_cost#theano.function(inputs, cost, name='f_cost',on_unused_input='ignore')
    f_pred = theano.function(inputs, enc_dec.decoder.argpred, \
                             name='f_pred',on_unused_input='ignore')

    #print enc_dec.params
    lr = tensor.scalar(name='lr')
    trainer = Trainer()

    logger.debug("Compiling grads")
    f_grad_shared, f_update = trainer.SetupTainer(lr,
                                inputs, cost, params)
    logger.debug("Compiling grads over")

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    kf_fix = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

    iter_epoches = len(train[0]) / batch_size
    if validFreq == -1:
        validFreq = 370 if iter_epoches < 370 else iter_epoches
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    # x is a list wrt a sentence
    rev_seq_f = lambda x: x[::-1]
    # x is a list wrt a sentence, for generating y from r2l
    rev_seq_f_eos = lambda x: x[:-1][::-1] + [x[-1]]
    # xx is a minibatch, i.e. a list of list
    reverse_lst_f = lambda xx: map(rev_seq_f,xx)
    reverse_lst_f_eos = lambda xx: map(rev_seq_f_eos,xx)

    print "validFreq, saveFreq",validFreq, saveFreq
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    model_f_iter, model_r_iter = None, None
    try:
        for eidx in xrange(max_epochs):
            tot_cost = 0
            n_samples = 0

            model_f_iter=saveto+'_f_iter%i.npz'%eidx
            model_r_iter=saveto+'_r_iter%i.npz'%eidx
            if model_options['save_on_the_fly']: 
                rev_enc_dec.save(model_r_iter)
                enc_dec.save(model_f_iter)

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            #kf = kf_fix[:1]
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)
                # Select the random examples for this minibatch
                xx = [train[0][t]for t in train_index]
                yy = [train[1][t] for t in train_index]
                #align = [train[2][t] for t in train_index]
                #label = [train[3][t] for t in train_index]

                x, x_mask, y, y_mask, _, _ = \
                    prepare_reorderdata_minibatch(xx, yy)
                n_samples += x.shape[1]

                r_x, r_x_m, r_y, r_y_m, _, _ = \
                    prepare_reorderdata_minibatch(reverse_lst_f(xx),reverse_lst_f_eos(yy))

                f_inputs = [x,x_mask,y,y_mask,r_x,r_x_m,r_y,r_y_m]
                b_time = time.time()
                cost = 0
                cost = f_grad_shared(*f_inputs)
                e_time = time.time()
                f_update(lrate)
                tot_cost += cost
                ## update after testing
                #f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.
                if numpy.mod(uidx, dispFreq) == 0:
                    #, gold_prob#'cost_bak', cost_bak
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    print '----------------l2r'
                    beamsearch.minibatch(x,x_mask,y,y_mask)
                    print '----------------r2l'
                    beamsearch_rev.minibatch(r_x,r_x_m,r_y,r_y_m)
                    logger.debug("Current speed (including update params) is {} per sentence for {} sentences".
                        format((e_time - b_time) / len(x[0]),len(x[0])))

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',
                    '''
                    if best_p is not None:
                        params = best_p
                    else:
                        params = enc_dec.unzip()
                    '''
                    enc_dec.save(saveto+'_f.npz',history_errs=history_errs)
                    rev_enc_dec.save(saveto+'_r.npz',history_errs=history_errs)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'


            print 'Seen %d samples, tot_cost %f' % (n_samples,tot_cost)
            if eidx+1 % 5 == 0:
                sys.stdout.flush()
            if  dev_file and eidx > begin_valid:
                if not os.path.isfile(model_f_iter):
                    print 'no model file=%s, finish decoding'%model_f_iter
                    continue;
                print 'evaluating on dev and test set for iter %s'%model_f_iter
                print_tparams(beamsearch.enc_dec.params)
                beamsearch.enc_dec.load(model_f_iter)
                print 'after load model'
                print_tparams(beamsearch.enc_dec.params)
                beamsearch_rev.enc_dec.load(model_r_iter)
                os.system('echo %s >>eval.txt'%model_f_iter)
                test_eval_combine(beamsearch,beamsearch_rev,src_file=dev_file,\
                                  trg_file=dev_ref,src_xml=dev_xml,modelfile=model_f_iter)
                if 1 and tst_file:
                    test_eval_combine(beamsearch,beamsearch_rev,src_file=tst_file,\
                              trg_file=tst_ref,src_xml=tst_xml,isdev=False,modelfile=model_f_iter)
            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    return #train_err, valid_err, test_err

def get_bleu(scorefile):
    lines = open(scorefile).readlines()
    m = re.search('bleu = (.+?), length_ratio', lines[0])
    if m:
        return float(m.group(1))
    else:
        die;

def split_underbar(line):
    """
    for target subword output like: r_o k_u 
    the expected output should be: r o k u for testing
    """
    subwords = line.split()
    res = []
    for s in subwords:
        #res += ' '.join(s.split('_'))
        res.append(' '.join(s.split('_')))
    return ' '.join(res)




def test_eval_combine(
    beamsearch,
    beamsearch_rev,
    src_file='srcfile',
    src_xml=None,
    trg_file='trgfile',
    src_dict='src.dict.pkl',
    trg_dict='trg.dict.pkl',
    isdev=True,
    modelfile=None,
):

    res = 'src.res.comb'
    res_f = 'src.res.forward'
    res_r = 'src.res.backward'
    res_rerank = 'src.res.rerank'
    ff = open(res,'w')
    ff_f = open(res_f,'w')
    ff_r = open(res_r,'w')
    ff_rerank = open(res_rerank,'w')
    start_time = time.time()
    for i, s in enumerate(open(src_file)):
        best_tran,best_f, best_r,best_rerank = combine(beamsearch,beamsearch_rev,s)
        best_tran = split_underbar(best_tran)
        best_f = split_underbar(best_f)
        best_r = split_underbar(best_r)
        best_rerank = split_underbar(best_rerank)

        if (i + 1)  % 100 == 0:
            ff.flush()
            logger.debug("Current speed is {} per sentence".
                    format((time.time() - start_time) / (i + 1)))
        if src_xml is None:
            print >>ff, best_tran
            print >>ff_f, best_f
            print >>ff_r, best_r
            print >>ff_rerank, best_rerank
        else:
            print >>ff,"%d ||| %s ||| 0.0 "%(i, best_tran)
            print >>ff_f,"%d ||| %s ||| 0.0 "%(i, best_f)
            print >>ff_r,"%d ||| %s ||| 0.0 "%(i, best_r)
            print >>ff_rerank,"%d ||| %s ||| 0.0 "%(i, best_rerank)

    logger.debug("Current speed is {} per sentence calculated on tot {} sentences".
         format((time.time() - start_time) / (i + 1),i+1))

    ff.close()
    ff_f.close()
    ff_r.close()
    ff_rerank.close()

    dev_or_tst = 'on dev set' if isdev else 'on tst set'

    print 'ACC for forward',dev_or_tst
    os.system('echo forward result >>eval.txt')
    evaluation(
        res=res_f,
        src_xml=src_xml,
        trg_file=trg_file,
        src_file=src_file,
        isdev=isdev,
        )

    print 'ACC for reverse',dev_or_tst
    os.system('echo backward result >>eval.txt')
    evaluation(
        res=res_r,
        src_xml=src_xml,
        trg_file=trg_file,
        src_file=src_file,
        isdev=isdev,
        )
    os.system('echo k-best approximation result >>eval.txt')
    evaluation(
        res=res_rerank,
        src_xml=src_xml,
        trg_file=trg_file,
        src_file=src_file,
        isdev=isdev,
        )



def combine(
    beamsearch,
    beamsearch_rev,
    src_sent, ### it is a word list
    combine_size=None,
):
    """ combine the results from the kbest of both beam searchers
        pair-wise combination with the latent intersection point.
    """
    if combine_size is None: combine_size = beamsearch.beamsize
    start_time = time.time()
    s = seq2indices(src_sent.strip().split(),beamsearch.s_word2index)
    #print 'src_sent',src_sent,s,type(src_sent)
    trans1,best_f = beamsearch.kbest_decode(s)
    trans1 = list(trans1)[:combine_size]
    #print beamsearch.prob_bisent_indice(s,list(trans1[0]))
    trans2,best_r = beamsearch_rev.kbest_decode(s)
    d_time = time.time()
    #logger.debug("Current search speed is {} per sentence".
    #    format((d_time - start_time) / 2))

    l = len(trans2[0])
    trans2 = list(trans2)[:combine_size]
    #print beamsearch_rev.prob_bisent_indice(s,list(trans2[0]))
    assert len(trans2[0]) == l, 'target is changed'
    heap = []
    trans_set = set()

    #print 'len(trans1)',len(trans1)
    #print 'len(trans2)',len(trans2)
    cands = []

    rerank_cands = list(trans1)
    rerank_cands.extend(trans2)
    rerank_cands = [list(x) for x in rerank_cands]
    cost_f_seq = []
    cost_r_seq = []
    for x in trans1:
        x = list(x)
        for y in trans2:
            y = list(y)
            for px in xrange(len(x)+1):
                for py in xrange(len(y)+1):
                    t1 = x[:px] + y[py:]
                    t2 = y[:py] + x[px:]
                    tt1 = tuple(t1)
                    tt2 = tuple(t2)
                    if tt1 not in trans_set:
                        trans_set.add(tt1)
                        cands.append(t1)

    e_time = time.time()
    heap_rerank = rescore(s,rerank_cands,beamsearch,beamsearch_rev)
    #print 'beam_size=%d heap_rerank.size=%d'%(beamsearch.beamsize,len(heap_rerank))
    best_rerank = heapq.nlargest(beamsearch.beamsize, heap_rerank)[0][1]
    best_rerank = beamsearch.to_words(best_rerank, beamsearch.t_index2word)
    heap = []
    batches = split2batches(cands)
    for bt in batches:
        pairs = rescore(s,bt,beamsearch,beamsearch_rev)
        heap.extend(pairs)

    kbest = heapq.nlargest(beamsearch.beamsize, heap)
    return beamsearch.to_words(kbest[0][1],beamsearch.t_index2word),best_f,best_r,best_rerank

def split2batches(
    cands,
    max_len=2000,
):
    batches = []
    size = len(cands)/max_len
    if len(cands)%max_len != 0:
        size += 1
    for i in xrange(size):
        start = i*max_len
        end = min((i+1)*max_len,len(cands))
        batches.append(cands[start:end])
    return batches

def rescore(
    s,
    cands,
    beamsearch,
    beamsearch_rev,
):
    xs = [s]*len(cands)
    # x is a list wrt a sentence
    rev_seq_f = lambda x: x[::-1]
    # x is a list wrt a sentence, for generating y from r2l
    rev_seq_f_eos = lambda x: x[::-1] + [beamsearch.eos_id]
    # xx is a minibatch, i.e. a list of list
    reverse_lst_f = lambda xx: map(rev_seq_f,xx)
    reverse_lst_f_eos = lambda xx: map(rev_seq_f_eos,xx)

    x, x_mask, y, y_mask, _, _ = \
         prepare_reorderdata_minibatch(reverse_lst_f(xs),cands)

    seq_f_eos = lambda x: x + [beamsearch.eos_id]
    add_seq_f_eos = lambda xx: map(seq_f_eos,xx)

    x, x_mask, y, y_mask, _, _ = \
         prepare_reorderdata_minibatch(reverse_lst_f(xs),add_seq_f_eos(cands))


    costs_f = beamsearch.enc_dec.f_sent_cost(x,x_mask,y,y_mask)
    #print 'forward: batch_cost',costs_f[:10]
    #print 'online cost',cost_f_seq[:10]

    r_x, r_x_m, r_y, r_y_m, _, _ = \
         prepare_reorderdata_minibatch(xs,reverse_lst_f_eos(cands))

    costs_r = beamsearch_rev.enc_dec.f_sent_cost(r_x,r_x_m,r_y,r_y_m)
    #f_inputs = [x,x_mask,y,y_mask,r_x,r_x_m,r_y,r_y_m]
    #print 'backward, batch_cost',costs_r[:10]
    #print 'online cost',cost_r_seq[:10]

    costs = -costs_f - costs_r
    return zip(costs,cands)


def evaluation(
    res='src.res',
    src_xml=None,
    trg_file='trgfile',
    src_file='srcfile',
    isdev=True,
):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    toxml = current_dir + '/scripts/kbest2xml.pl'
    evaluator = current_dir + '/scripts/news_evaluation.py'

    res_xml = '%s.xml'%res
    print 'perl %s %s %s %s'%(toxml,src_xml,res,res_xml)
    os.system('perl %s %s %s %s'%(toxml,src_xml,res,res_xml))
    if isdev:
        os.system('echo dev >>eval.txt; cp %s dev.%s'%(res,res))
    else:
        os.system('echo tst >>eval.txt; cp %s tst.%s'%(res,res))
    cmd = 'perl %s -t %s -i %s -o eval.log >acc.log; ' %(evaluator, trg_file,res_xml) + \
          'cat acc.log; cat acc.log >>eval.txt'
    print cmd
    os.system(cmd)


def prob_calculater(
    beamsearch = None,
    s_seq = [],
    t_seq = [],
):
    s_seq = numpy.array(s_seq)[:,None]
    t_seq = numpy.array(t_seq)[:,None]
    s_mask = numpy.ones(s_seq.shape[0], dtype=config.floatX)[:,None]
    t_mask = numpy.ones(t_seq.shape[0], dtype=config.floatX)[:,None]
    #beamsearch.minibatch(s_seq,s_mask,t_seq,t_mask)
    return beamsearch.enc_dec.f_cost(s_seq,s_mask,t_seq,t_mask)

def decoding_check(
    beamsearch=None,
    src_file='srcfile',
    trg_file='trgfile',
):
    from itertools import izip
    for i, (line,t_line) in enumerate(izip(open(src_file),open(trg_file))):
        score,best_tran = beamsearch.decode(line)
        ref_score = beamsearch.prob_bisent(line,t_line)
        print "sent id %d, ref_score=%f, score=%f"%(i,ref_score,score)
        print "src: %s\ntrg: %s\nref: %s"%(line.strip(),best_tran,t_line.strip())

def get_train_opt(file):
    def fn_config(
        max_epochs=100,
        dim_proj=500,
        layers=1,
        begin_valid=0,
        src_file=None,
        trg_file=None,
        align_file=None,
        patience=100,
        batch_size=16,
        dev_file=None,
        dev_ref=None,
        dev_xml=None,
        tst_file=None,
        tst_ref=None,
        tst_xml=None,
        reload_model=None,
        bi_reg=None,
    ):
        return locals().copy()

    config = fn_config()

    for line in open(file):
        fields = line.strip().split()
        if len(fields) < 3:
            continue
        if fields[0] != 'int' and fields[0] != 'str' and fields[0] != 'float':
            continue
        type, k, v = fields[0], fields[1], fields[2]
        config[k] = eval(type)(v)
        print type, k, v
    assert config["src_file"] is not None, 'reset src_file'

    train_lstm(**config)




if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s: %(filename)s:%(lineno)s - %(funcName)s: %(levelname)s: %(message)s")

    dir = "/panfs/panmt/users/lliu/code/lstm_reorder/h10k-iwslt-data"
    src = "%s/h10k.zh"%dir
    trg = "%s/h10k.en"%dir
    align = "%s/h10k.align"%dir


    get_train_opt(sys.argv[1])

    sys.exit()

    devdir = '/panfs/panmt/users/lliu/data/iwslt/iwslt2009/task_BETC_CT/dev'
    dev_src = '%s/IWSLT09.devset1_CSTAR03.zh'%devdir
    dev_ref = '%s/IWSLT09.devset1_CSTAR03.low.ref'%devdir
    tst_src = '%s/2004/IWSLT09.devset2_IWSLT04.zh'%devdir
    tst_ref = '%s/2004/IWSLT09.devset2_IWSLT04.low.ref'%devdir
    tst_xml=''

    dev_src = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/corpora/test/NEWS10_test_JaEn_1935.txt'
    dev_xml = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/corpora/test/NEWS10_test_JaEn_1935.xml'
    dev_ref = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/corpora/test/NEWS10_ref_JaEN_1935.xml'

    src = '/panfs/panmt/users/lliu/data/iwslt/iwslt2009/task_BETC_CT/training-data/IWSLT09_BTEC_CT.clean.zh'
    trg = '/panfs/panmt/users/lliu/data/iwslt/iwslt2009/task_BETC_CT/training-data/IWSLT09_BTEC_CT.clean.en'
    align = '/panfs/panmt/users/lliu/data/iwslt/iwslt2009/task_BETC_CT/training-data/hiero-table/model/aligned.grow-diag-final-and'


    src = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/phrase_alignment/word_data/train.ja.txt'
    trg = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/phrase_alignment/word_data/train.en.txt'
    #trg = '/panfs/panmt/users/lliu/data/transliteration/NEWS2012/corpora/test/NEWS10_ref_JaEN_1935.txt'
    #trg = dev_ref

    '''
    src = "%s/h5h.zh"%dir
    trg = "%s/h5h.en"%dir
    align = "%s/h5h.align"%dir

    src = "%s/h2.zh"%dir
    trg = "%s/h2.en"%dir
    align = "%s/h2.align"%dir

    src = "%s/h60.zh"%dir
    trg = "%s/h60.en"%dir
    align = "%s/h60.align"%dir

    src = "%s/h2.zh"%dir
    trg = "%s/h2.en"%dir
    align = "%s/h2.align"%dir

    '''
    train_lstm(
        max_epochs=1000,
        dim_proj=500,
        layers=1,
        test_size=500,
        src_file=src,
        trg_file=trg,
        align_file=align,
        patience=100,
        batch_size=16,  # The batch size during training.
        dev_file=dev_src,
        dev_ref=dev_ref,
        dev_xml=dev_xml,
        tst_file=tst_src,
        tst_ref=tst_ref,
        tst_xml=tst_xml,
    )
    '''
    decoding_check(
        src_file=src,
    )
    '''
