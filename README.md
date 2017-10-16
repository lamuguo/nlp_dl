
Deep Learning NLP Pipeline implemented on Tensorflow. Following the 'simplicity' rule, this project aims to 
use the deep learning library of Tensorflow to implement new NLP pipeline. You can extend the project to 
train models with your own corpus/languages. Pretrained models of Chinese corpus are distributed.

Brief Introduction
==================
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
