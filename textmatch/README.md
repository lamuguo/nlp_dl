# textmatch
----
textmatch providing function for text matching. It was developed with a focus on facilitate the designing, comparing and sharing of deep text matching models. 

## Overview
There are three modules in this tool, including namely data preparation, model construction, training and evaluation, respectively, which organized as a pipeline of data flow.

### Data Preparation
The tool could convert dataset of different text matching tasks into a unified format as the input of deep matching models.

### Training
- a variety of objective functions for regression, classification and ranking
- point-wise,pair-wise,list-wise losses

## Implemented Models
- [DRMM](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)
- [MatchPyramid](https://arxiv.org/abs/1602.06359)
- [ARC-I](https://arxiv.org/abs/1503.03244)
- [DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
- [CDSSM](https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/)
- [ARC-II](https://arxiv.org/abs/1503.03244)
- [Match_SRNN](https://www.ijcai.org/Proceedings/16/Papers/415.pdf)
- [Siamese_LSTM]() [Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W16-16#page=162),[Siamese Recurrent Architectures for Learning Sentence Similarity](Siamese Recurrent Architectures for Learning Sentence Similarity)
## Usage
```
git clone https://github.com/tangzhenyu/nlp_dl/textmatch.git
cd textmatch
python setup.py install

python main.py --phase train --model_file ./models/drmm.config
python main.py --phase predict --model_file ./models/drmm.config
```
## Dependency
* python2.7+ 
* tensorflow 1.2+
* keras 2.0+
