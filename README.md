
Deep Learning for NLP,implemente some tools,like pos,ner,textsum on Tensorflow. Following the 'simplicity' rule, this project aims to 
use the deep learning library of Tensorflow to implement new NLP pipeline. You can extend the project to 
train models with your own corpus/languages. Pretrained models of Chinese corpus are distributed.


Brief Introduction
==================
* [Papers](#papers)
* [Modules](#modules)
* [Installation](#installation)
* [Tutorial](#tutorial)
    * [Segmentation](#segmentation)
    * [POS](#pos)
    * [NER](#ner)
    * [Pipeline](#pipeline)
    * [Textsum](#textsum)
    * [Textcnn](#textcnn)
    * [Train your model](#train-your-model)
* [Reference](#reference)

Papers
========
### Text Summarization
  - Ryang, Seonggi, and Takeshi Abekawa. "[Framework of automatic text summarization using reinforcement learning](http://dl.acm.org/citation.cfm?id=2390980)." In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 256-265. Association for Computational Linguistics, 2012. [not neural-based methods]
  - King, Ben, Rahul Jha, Tyler Johnson, Vaishnavi Sundararajan, and Clayton Scott. "[Experiments in Automatic Text Summarization Using Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8775&rep=rep1&type=pdf)." Machine Learning (2011).
  - Liu, Yan, Sheng-hua Zhong, and Wenjie Li. "[Query-Oriented Multi-Document Summarization via Unsupervised Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5058/5322)." AAAI. 2012.
  - Rioux, Cody, Sadid A. Hasan, and Yllias Chali. "[Fear the REAPER: A System for Automatic Multi-Document Summarization with Reinforcement Learning](http://emnlp2014.org/papers/pdf/EMNLP2014075.pdf)." In EMNLP, pp. 681-690. 2014.[not neural-based methods]
  - PadmaPriya, G., and K. Duraiswamy. "[An Approach For Text Summarization Using Deep Learning Algorithm](http://thescipub.com/PDF/jcssp.2014.1.9.pdf)." Journal of Computer Science 10, no. 1 (2013): 1-9.
  - Denil, Misha, Alban Demiraj, and Nando de Freitas. "[Extraction of Salient Sentences from Labelled Documents](http://arxiv.org/abs/1412.6815)." arXiv preprint arXiv:1412.6815 (2014).
  - Kågebäck, Mikael, et al. "[Extractive summarization using continuous vector space models](http://www.aclweb.org/anthology/W14-1504)." Proceedings of the 2nd Workshop on Continuous Vector Space Models and their Compositionality (CVSC)@ EACL. 2014.
  - Denil, Misha, Alban Demiraj, Nal Kalchbrenner, Phil Blunsom, and Nando de Freitas. "[Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network](http://arxiv.org/abs/1406.3830)." arXiv preprint arXiv:1406.3830 (2014).
  - Cao, Ziqiang, Furu Wei, Li Dong, Sujian Li, and Ming Zhou. "[Ranking with Recursive Neural Networks and Its Application to Multi-document Summarization](http://gana.nlsde.buaa.edu.cn/~lidong/aaai15-rec_sentence_ranking.pdf)." (AAAI'2015).
  - Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, and Noah A. Smith. "[Toward Abstractive Summarization Using Semantic Representations](http://www.cs.cmu.edu/~nasmith/papers/liu+flanigan+thomson+sadeh+smith.naacl15.pdf)." NAACL 2015
  - Wenpeng Yin， Yulong Pei. "Optimizing Sentence Modeling and Selection for Document Summarization." IJCAI 2015
  - He, Zhanying, Chun Chen, Jiajun Bu, Can Wang, Lijun Zhang, Deng Cai, and Xiaofei He. "[Document Summarization Based on Data Reconstruction](http://cs.nju.edu.cn/zlj/pdf/AAAI-2012-He.pdf)." In AAAI. 2012.
  - Liu, He, Hongliang Yu, and Zhi-Hong Deng. "[Multi-Document Summarization Based on Two-Level Sparse Representation Model](http://www.cis.pku.edu.cn/faculty/system/dengzhihong/papers/AAAI%202015_Multi-Document%20Summarization%20Based%20on%20Two-Level%20Sparse%20Representation%20Model.pdf)." In Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  - Jin-ge Yao, Xiaojun Wan, Jianguo Xiao. "[Compressive Document Summarization via Sparse Optimization](http://ijcai.org/Proceedings/15/Papers/198.pdf)." IJCAI 2015
  - Piji Li, Lidong Bing, Wai Lam, Hang Li, and Yi Liao. "[Reader-Aware Multi-Document Summarization via Sparse Coding](http://arxiv.org/abs/1504.07324)." IJCAI 2015.
  - Lopyrev, Konstantin. "[Generating News Headlines with Recurrent Neural Networks](http://arxiv.org/abs/1512.01712)." arXiv preprint arXiv:1512.01712 (2015). [The first paragraph as document.]
  - Alexander M. Rush, Sumit Chopra, Jason Weston. "[A Neural Attention Model for Abstractive Sentence Summarization](http://arxiv.org/abs/1509.00685)." EMNLP 2015. [sentence compression]
  - Hu, Baotian, Qingcai Chen, and Fangze Zhu. "[LCSTS: a large scale chinese short text summarization dataset](http://arxiv.org/abs/1506.05865)." arXiv preprint arXiv:1506.05865 (2015).
  - Gulcehre, Caglar, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. "[Pointing the Unknown Words](http://arxiv.org/abs/1603.08148)." arXiv preprint arXiv:1603.08148 (2016).
  - Nallapati, Ramesh, Bing Xiang, and Bowen Zhou. "[Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](http://arxiv.org/abs/1602.06023)." arXiv preprint arXiv:1602.06023 (2016). [sentence compression]
  - Sumit Chopra, Alexander M. Rush and Michael Auli. "[Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://harvardnlp.github.io/papers/naacl16_summary.pdf)" NAACL 2016.
  - Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. "[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/abs/1603.06393)." ACL. (2016)
  - Jianpeng Cheng, Mirella Lapata. "[Neural Summarization by Extracting Sentences and Words](http://arxiv.org/abs/1603.07252)". ACL. (2016)
  - Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. "[AttSum: Joint Learning of Focusing and Summarization with Neural Attention](http://arxiv.org/abs/1604.00125)".  arXiv:1604.00125 (2016)
  - Ayana, Shiqi Shen, Zhiyuan Liu, Maosong Sun. "[Neural Headline Generation with Sentence-wise Optimization](http://arxiv.org/abs/1604.01904)". arXiv:1604.01904 (2016)
  - Kikuchi, Yuta, Graham Neubig, Ryohei Sasano, Hiroya Takamura, and Manabu Okumura. "[Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552)." arXiv preprint arXiv:1609.09552 (2016).
  - Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei and Hui Jiang. "[Distraction-Based Neural Networks for Document Summarization](https://arxiv.org/abs/1610.08462)." IJCAI 2016.
  - Wang, Lu, and Wang Ling. "[Neural Network-Based Abstract Generation for Opinions and Arguments](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf)." NAACL 2016.
  - Yishu Miao, Phil Blunsom. "[Language as a Latent Variable: Discrete Generative Models for Sentence Compression](http://arxiv.org/abs/1609.07317)." EMNLP 2016.
  - Hongya Song, Zhaochun Ren, Piji Li, Shangsong Liang, Jun Ma, and Maarten de Rijke. [Summarizing Answers in Non-Factoid Community Question-Answering](). In WSDM 2017: The 10th International Conference on Web Search and Data Mining, 2017.
  - Wenyuan Zeng, Wenjie Luo, Sanja Fidler, Raquel Urtasun. "[Efficient Summarization with Read-Again and Copy Mechanism](https://arxiv.org/abs/1611.03382)." arXiv preprint arXiv:1611.03382 (2016).
  - Piji Li, Zihao Wang, Wai Lam, Zhaochun Ren, Lidong Bing. Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization. In AAAI, 2017.
  - Ramesh Nallapati, Feifei Zhai, Bowen Zhou. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230). In AAAI, 2017.
  - Ramesh Nallapati, Bowen Zhou, Mingbo Ma. "[Classify or Select: Neural Architectures for Extractive Document Summarization](https://arxiv.org/abs/1611.04244)." arXiv preprint arXiv:1611.04244 (2016).

### Text Matching
  - [DeepRank](https://arxiv.org/abs/1710.05649)
  - [Match-SRNN](https://arxiv.org/abs/1604.04378)
  - []()<a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>
  - []()<a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>
  - []()<a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
  - []()<a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>
  - <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>
  - <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
  - <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>
  - <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
  - <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

### Opinion Summarization
  - Wu, Haibing, Yiwei Gu, Shangdi Sun, and Xiaodong Gu. "[Aspect-based Opinion Summarization with Convolutional Neural Networks](http://arxiv.org/abs/1511.09128)." arXiv preprint arXiv:1511.09128 (2015).
  - Irsoy, Ozan, and Claire Cardie. "[Opinion Mining with Deep Recurrent Neural Networks](http://anthology.aclweb.org/D/D14/D14-1080.pdf)." In EMNLP, pp. 720-728. 2014.

### Reading Comprehension
 - Hermann, Karl Moritz, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. "[Teaching machines to read and comprehend](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend)." In Advances in Neural Information Processing Systems, pp. 1693-1701. 2015.
 - Hill, Felix, Antoine Bordes, Sumit Chopra, and Jason Weston. "[The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](http://arxiv.org/abs/1511.02301)." arXiv preprint arXiv:1511.02301 (2015).
 - Kadlec, Rudolf, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. "[Text Understanding with the Attention Sum Reader Network](http://arxiv.org/abs/1603.01547)." arXiv preprint arXiv:1603.01547 (2016).
 - Chen, Danqi, Jason Bolton, and Christopher D. Manning. "[A thorough examination of the cnn/daily mail reading comprehension task](http://arxiv.org/abs/1606.02858)." arXiv preprint arXiv:1606.02858 (2016).
 - Dhingra, Bhuwan, Hanxiao Liu, William W. Cohen, and Ruslan Salakhutdinov. "[Gated-Attention Readers for Text Comprehension](http://arxiv.org/abs/1606.01549)." arXiv preprint arXiv:1606.01549 (2016).
 - Sordoni, Alessandro, Phillip Bachman, and Yoshua Bengio. "[Iterative Alternating Neural Attention for Machine Reading](http://arxiv.org/abs/1606.02245)." arXiv preprint arXiv:1606.02245 (2016).
 - Trischler, Adam, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. "[Natural Language Comprehension with the EpiReader](http://arxiv.org/abs/1606.02270)." arXiv preprint arXiv:1606.02270 (2016).
 - Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, Guoping Hu. "[Attention-over-Attention Neural Networks for Reading Comprehension](http://arxiv.org/abs/1607.04423)." arXiv preprint arXiv:1607.04423 (2016).
 - Yiming Cui, Ting Liu, Zhipeng Chen, Shijin Wang, Guoping Hu. "[Consensus Attention-based Neural Networks for Chinese Reading Comprehension](https://arxiv.org/abs/1607.02250)." arXiv preprint arXiv:1607.02250 (2016).
 - Daniel Hewlett, Alexandre Lacoste, Llion Jones, Illia Polosukhin, Andrew Fandrianto, Jay Han, Matthew Kelcey and David Berthelot. "[WIKIREADING: A Novel Large-scale Language Understanding Task over Wikipedia](http://www.aclweb.org/anthology/P/P16/P16-1145.pdf)." ACL (2016). pp. 1535-1545.
  

### Sentence Modelling
  - Kalchbrenner, Nal, Edward Grefenstette, and Phil Blunsom. "[A convolutional neural network for modelling sentences](http://arxiv.org/abs/1404.2188)." arXiv preprint arXiv:1404.2188 (2014).
  - Kim, Yoon. "[Convolutional neural networks for sentence classification](http://arxiv.org/abs/1408.5882)." arXiv preprint arXiv:1408.5882 (2014).
  - Le, Quoc V., and Tomas Mikolov. "[Distributed representations of sentences and documents](http://arxiv.org/abs/1405.4053)." arXiv preprint arXiv:1405.4053 (2014).
  - Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. "[Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)." In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.

### Reasoning
  - Peng, Baolin, Zhengdong Lu, Hang Li, and Kam-Fai Wong. "[Towards Neural Network-based Reasoning](http://arxiv.org/abs/1508.05508)." arXiv preprint arXiv:1508.05508 (2015).
  
### Knowledge Engine
 - Bordes, Antoine, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. "[Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)." In Advances in Neural Information Processing Systems, pp. 2787-2795. 2013. TransE
 - Lin, Yankai, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. "[Neural Relation Extraction with Selective Attention over Instances](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2016_nre.pdf)." ACL (2016)
 - TransXXX

### Memory Networks
 - Graves, Alex, Greg Wayne, and Ivo Danihelka. "[Neural turing machines](http://arxiv.org/abs/1410.5401)." arXiv preprint arXiv:1410.5401 (2014).
 - Weston, Jason, Sumit Chopra, and Antoine Bordes. "[Memory networks](http://arxiv.org/abs/1410.3916)." ICLR (2014).
 - Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "[End-to-end memory networks](http://papers.nips.cc/paper/5846-end-to-end-memory-networks)." In Advances in neural information processing systems, pp. 2440-2448. 2015.
 - Weston, Jason, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, and Tomas Mikolov. "[Towards ai-complete question answering: A set of prerequisite toy tasks](http://arxiv.org/abs/1502.05698)." arXiv preprint arXiv:1502.05698 (2015).
 - Bordes, Antoine, Nicolas Usunier, Sumit Chopra, and Jason Weston. "[Large-scale simple question answering with memory networks](http://arxiv.org/abs/1506.02075)." arXiv preprint arXiv:1506.02075 (2015).
 - Kumar, Ankit, Ozan Irsoy, Jonathan Su, James Bradbury, Robert English, Brian Pierce, Peter Ondruska, Ishaan Gulrajani, and Richard Socher. "[Ask me anything: Dynamic memory networks for natural language processing](http://arxiv.org/abs/1506.07285)." arXiv preprint arXiv:1506.07285 (2015).
 - Dodge, Jesse, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, and Jason Weston. "[Evaluating prerequisite qualities for learning end-to-end dialog systems](http://arxiv.org/abs/1511.06931)." arXiv preprint arXiv:1511.06931 (2015).
 - Hill, Felix, Antoine Bordes, Sumit Chopra, and Jason Weston. "[The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](http://arxiv.org/abs/1511.02301)." arXiv preprint arXiv:1511.02301 (2015).
 - Weston, Jason. "[Dialog-based Language Learning](http://arxiv.org/abs/1604.06045)." arXiv preprint arXiv:1604.06045 (2016).
 - Bordes, Antoine, and Jason Weston. "[Learning End-to-End Goal-Oriented Dialog](http://arxiv.org/abs/1605.07683)." arXiv preprint arXiv:1605.07683 (2016).
 - Chandar, Sarath, Sungjin Ahn, Hugo Larochelle, Pascal Vincent, Gerald Tesauro, and Yoshua Bengio. "[Hierarchical Memory Networks](https://arxiv.org/abs/1605.07427)." arXiv preprint arXiv:1605.07427 (2016).
 - Jason Weston."[Memory Networks for Language Understanding](http://www.thespermwhale.com/jaseweston/icml2016/)." ICML Tutorial 2016
 - Tang, Yaohua, Fandong Meng, Zhengdong Lu, Hang Li, and Philip LH Yu. "[Neural Machine Translation with External Phrase Memory](http://arxiv.org/abs/1606.01792)." arXiv preprint arXiv:1606.01792 (2016).
 - Wang, Mingxuan, Zhengdong Lu, Hang Li, and Qun Liu. "[Memory-enhanced Decoder for Neural Machine Translation](http://arxiv.org/abs/1606.02003)." arXiv preprint arXiv:1606.02003 (2016).
 - Xiong, Caiming, Stephen Merity, and Richard Socher. "[Dynamic memory networks for visual and textual question answering](https://arxiv.org/abs/1603.01417)." arXiv preprint arXiv:1603.01417 (2016).

Modules
========
* NLP Pipeline Modules:
    * Word Segmentation/Tokenization
    * Part-of-speech (POS)
    * Named-entity-recognition(NER)
    * textsum: automatic summarization Seq2Seq-Attention models
    * textcnn: document classification
    * Web API: Free Tensorflow empowered web API
    * Planed: Parsing, Automatic Summarization

* Algorithm(Closely following the state-of-Art)
    * Word Segmentation: Linear Chain CRF(conditional-random-field), based on python CRF++ module
    * POS: LSTM/BI-LSTM network, based on Tensorflow
    * NER: LSTM/BI-LSTM/LSTM-CRF network, based on Tensorflow
    * Textsum: Seq2Seq with attention mechanism
    * Texncnn: CNN

* Pre-trained Model
    * Chinese: Segmentation, POS, NER (1998 china daily corpus)
    * English: POS (brown corpus)
    * For your Specific Language, you can easily use the script to train model with the corpus of your language choice.

Installation
================
* Requirements
    * CRF++ (>=0.54)
    * Tensorflow(1.0) 
This project is up to date with the latest tensorflow release. 

Due to pkg size restriction, english pos model, ner model files are not distributed on pypi
You can download the pre-trained model files from github and put in your installation directory .../site-packages/.../nlp_dl/...
model files: ../pos/ckpt/en/pos.ckpt  ; ../ner/ckpt/zh/ner.ckpt

* Running Examples
```python
    # ./ folder
    cd test
    python test_pos_en.py
    python test_segmenter.py
    python test_pos_zh.py
    python test_api_v1_module.py
    python test_api_v1_pipeline.py
```

Tutorial
===========
Set Coding
---------------
For python2, the default coding is ascii not unicode, use __future__ module to make it compatible with python3
```python
#coding=utf-8
from __future__ import unicode_literals # compatible with python3 unicode

```

Download pretrained models
---------------
If you install nlp_dl via pip, the pre-trained models are not distributed due to size restriction. 
You can download full models for 'Segment', 'POS' en and zh, 'NER' zh, 'Textsum' by calling the download function.

```python
import nlp_dl
# Download all the modules
nlp_dl.download()

# Download only specific module
nlp_dl.download('segment')
nlp_dl.download('pos')
nlp_dl.download('ner')
nlp_dl.download('textsum')
```

Segmentation
---------------
```python
#coding=utf-8
from __future__ import unicode_literals

from nlp_dl import segmenter

text = "我刚刚在浙江卫视看了电视剧老九门，觉得陈伟霆很帅"
segList = segmenter.seg(text)
text_seg = " ".join(segList)

print (text.encode('utf-8'))
print (text_seg.encode('utf-8'))

#Results
#我 刚刚 在 浙江卫视 看 了 电视剧 老九门 ， 觉得 陈伟霆 很 帅

```

POS
-----
```python
#coding:utf-8
from __future__ import unicode_literals

import nlp_dl
nlp_dl.download('pos')

## English Model
from nlp_dl import pos_tagger
tagger = pos_tagger.load_model(lang = 'en')  # Loading English model, lang code 'en', English Model Brown Corpus

text = "I want to see a funny movie"
words = text.split(" ")     # unicode
print (" ".join(words).encode('utf-8'))

tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))
    
#Results
#I/nn want/vb to/to see/vb a/at funny/jj movie/nn

## Chinese Model
from nlp_dl import segmenter
from nlp_dl import pos_tagger
tagger = pos_tagger.load_model(lang = 'zh') # Loading Chinese model, lang code 'zh', China Daily Corpus

text = "我爱吃北京烤鸭"
words = segmenter.seg(text) # words in unicode coding
print (" ".join(words).encode('utf-8'))

tagging = tagger.predict(words)  # input: unicode coding
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/r 爱/v 吃/v 北京/ns 烤鸭/n

```

NER
-----
```python
#coding:utf-8
from __future__ import unicode_literals

# Download pretrained NER model
import nlp_dl
nlp_dl.download('ner')

from nlp_dl import segmenter
from nlp_dl import ner_tagger
tagger = ner_tagger.load_model(lang = 'zh') # Loading Chinese NER model

text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print (" ".join(words).encode('utf-8'))

tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

#Results
#我/nt 爱/nt 吃/nt 北京/p 烤鸭/nt

```

Pipeline
----------
```python
#coding:utf-8
from __future__ import unicode_literals

from nlp_dl import pipeline
p = pipeline.load_model('zh')

#Segmentation
text = "我爱吃北京烤鸭"
res = p.analyze(text)

print (res[0].encode('utf-8'))
print (res[1].encode('utf-8'))
print (res[2].encode('utf-8'))

words = p.segment(text)
pos_tagging = p.tag_pos(words)
ner_tagging = p.tag_ner(words)

print (pos_tagging.encode('utf-8'))
print (ner_tagging.encode('utf-8'))

```

Textsum
---------------
See details: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/textsum)


TextCNN
---------------
See details: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/textcnn)

Train your model
--------------------
###Segment model
See instructions: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/segment)

###POS model
See instructions: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/pos)

###NER model
See instructions: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/ner)

###Textsum model
See instructions: [README](https://github.com/tangzhenyu/nlp_dl/tree/master/nlp_dl/textsum)


Reference
=======
* CRF++ package: 
https://taku910.github.io/crfpp/#download
* Tensorflow: 
https://www.tensorflow.org/
