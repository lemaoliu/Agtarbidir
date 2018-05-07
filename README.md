
![Janus logo](<img width="150" height="150" src="https://cloud.githubusercontent.com/assets/9007148/12938904/e25b103a-cffc-11e5-8a29-709a88c9550e.png"/>)


# JANUS - a Joint Agreement Neural transdUction System for sequence2sequence learning

This toolkit implements an agreement model between left-to-right and right-to-left recurrent neural networks (with LSTM hiddent units), which converts an input sequence to an output sequence.
It is mainly developed by [Lemao Liu](https://sites.google.com/site/lemaoliu/) @ [NICT](http://www.nict.go.jp/en/univ-com/) Japan, following the paper [1].

Currently, it supports:
- A single left to right model;
- A single right to left model;
- The agreement model between a single left-to-right and a single right-to-left model.

In order to get more powerful performance as shown in our paper, one might need the agreement model between the ensembles of left-to-right and right-to-left models, which will be available soon (on going). 


# Dependencies
- [Python2.7](https://www.python.org/download/releases/2.7/)
- [Theano](https://github.com/Theano/Theano)
- [Numpy](http://www.numpy.org/)
- XML::Simple and XML::Twig Perl Modules -
  It is only used for a evaluation script in performance. To to install it, one can use cpan. Of course, one shoule install   Perl (with Perl 5.10 verified) at first.



# Training
```
python nn.py config.ini
```
In the current directory, a sample of config.ini is available.
Generally, to obtain the *.xml files as shown in config.ini, please use scripts/wrapper_src_trg.py:
```
python scripts/wrapper_src_trg.py dev.ja dev.en >dev.ref.ja-en.xml 2>dev.src.ja-en.xml
```

# Testing
After the training, there will be a file (eval.txt) in the working directory. 
This file contains four evaluation metrics (ACC, Mean-F-score, MRR and MAP_ref) for three models (left-to-right, right-to-left, and their agreement models) at each iteration, and some lines in this file are as follows:
```
evaluate for 66 iterations
-------------
Performance for left-to-right model
dev
ACC:          0.270000
Mean F-score: 0.779298
MRR:          0.270000
MAP_ref:      0.270000
Performance for right-to-left model
dev
ACC:          0.460000
Mean F-score: 0.832969
MRR:          0.460000
MAP_ref:      0.460000
Performance for agreement model
dev
ACC:          0.630000
Mean F-score: 0.912477
MRR:          0.630000
MAP_ref:      0.630000
Performance for left-to-right model
tst
ACC:          0.260000
Mean F-score: 0.767345
MRR:          0.260000
MAP_ref:      0.260000
Performance for right-to-left model
tst
ACC:          0.450000
Mean F-score: 0.811715
MRR:          0.450000
MAP_ref:      0.450000
Performance for agreement model
tst
ACC:          0.690000
Mean F-score: 0.901153
MRR:          0.690000
MAP_ref:      0.690000
```
Note that to get the real testing performance, one should select the best iteration according the performance on the dev set and then report its corresponding testing performance on the test set.

# About the JANUS and its LOGO
Both the name and its logo are given by [Andrew Finch](http://www.andrewfinch.com/)


# Any bugs or questions
Please feel free to contact lemaoliu@gmail.com. Your feedbacks are invaluable to make the toolkit powerful. Thanks!

# Acknowledgement
Thanks to the tutorial at http://deeplearning.net/tutorial/lstm.html, where we follow the implementaion.

# References
[1]. Lemao Liu, Andrew Finch, Masao Utiyama, Eiichiro Sumita. [Agreement on Target-bidirectional LSTMs for Sequence-to-Sequence Learning](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxsZW1hb2xpdXxneDo0ZTdmOWJlN2U3ZDAwMDFi). In Proceedings of AAAI, 2016.



