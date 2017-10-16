Catalogue
==================
* [Tutorial](#tutorial)
    * [Corpus and Config](#corpus-and-config)
    * [Prediction](#prediction)
    * [Attention Visualization](#attention-visualization)
    * [Training](#training)
* [Reference](#reference)`

Tutorial
=================
This textsum example is implementing Seq2Seq model on tensorflow for the automatic summarization task.
The code is modified from the original English-French translate.py model in tensorflow tutorial. Evaluation method of ROUGE and
BLEU is provided in the eval.py module. Pre-trained Chinese news articles' headline generation model is also distributed.

Corpus and Config
-------------------
We choose the Chinese news corpus form sohu.com. You can download it from http://www.sogou.com/labs/resource/cs.php

Seq2Seq-attention config params:

Encoder LSTM:
*   num_layers = 4  # 4 layer LSTM
*   size = 256      # 256 nodes per layer
*   num_samples = 4096  # negative sampling during softmax 4096
*   batch_size = 64     # 64 examples per batch
*   vocab_size = 50000  # top 50000 words in dictionary

Bucket桶:
News article cutted to 120, news with fewer words will be padded with 'PAD', Titles length cut to 30.
*buckets = [(120, 30), ...]

Model File:
The pre-trained model file has name 'headline_large.ckpt-48000.data-00000-of-00001' and it
is compressed and split into 3 files: *.tar.gz00, *.tar.gz01, *.tar.gz02

```shell
#Model File Name: headline_large.ckpt-48000.data-00000-of-00001

cd ./ckpt
cat headline_large.ckpt-48000.* > headline_large.ckpt-48000.data-00000-of-00001.tar.gz
tar xzvf headline_large.ckpt-48000.data-00000-of-00001.tar.gz
```

Prediction
---------------
### Run the script
Run the predict.py and interactively input the Chinese news text(space separated), automatic generated headline will be returned.
```shell
python predict.py
```

Output examples
```shell
news:      中央 气象台 TAG_DATE TAG_NUMBER 时 继续 发布 暴雨 蓝色 预警 TAG_NAME_EN 预计 TAG_DATE TAG_NUMBER 时至 TAG_DATE TAG_NUMBER 时 TAG_NAME_EN 内蒙古 东北部 、 山西 中 北部 、 河北 中部 和 东北部 、 京津 地区 、 辽宁 西南部 、 吉林 中部 、 黑龙江 中部 偏南 等 地 的 部分 地区 有 大雨 或 暴雨 。
headline:  中央 气象台 发布 暴雨 蓝色 预警 华北 等 地 持续 暴雨

news:      美国 科罗拉多州 山林 大火 持续 肆虐 TAG_NAME_EN 当地 时间 TAG_DATE 横扫 州 内 第二 大 城市 科罗拉多斯 普林斯 一 处 居民区 TAG_NAME_EN 迫使 超过 TAG_NUMBER TAG_NAME_EN TAG_NUMBER 万 人 紧急 撤离 。 美国 正 值 山火 多发 季 TAG_NAME_EN 现有 TAG_NUMBER 场 山火 处于 活跃 状态 。
headline:  美国 多地 山火 致 TAG_NUMBER 人 死亡
...
```

### Evaluate ROUGE-score
Run the command line: python predict.py arg1 arg2 arg3;
args1: directory of space separated Chinese news content corpus, one article content per line;
args2: directory of human generated headline for the news, one title per line;
args3: directory of machine generated summary to be saved.

The model will call the evaluate(X, Y, method = "rouge_n", n = 2) method in the eval.py

```shell
folder_path=`pwd`
input_dir=${folder_path}/news/test/content-test.txt
reference_dir=${folder_path}/news/test/title-test.txt
summary_dir=${folder_path}/news/test/summary.txt

python predict.py $input_dir $reference_dir $summary_dir

```

Attention Visualization
---------------
To get the attention mask matrix, we need to modified the standard seq2seq ops tf.nn.seq2seq. Right now 
there is not available method to extract those tensors so we need to modify the source file. We save the modified file to seq2seq_attn.py in this package.

### Run predict_attn.py
```shell
# Call the method in eval.py: plot_attention(data, X_label=None, Y_label=None), based on matplotlib package
# The attention heatmap will be saved under the /img folder
python predict_attn.py

```
Examples:
![image](https://raw.githubusercontent.com/tangzhenyu/DeepLearning_NLP_Tool/master/nlp_dl/textsum/img/attention_heatmap_1488962653.jpg)

Training
---------------
### Corpus format
Prepare four documents: content-train.txt, title-train.txt, content-dev.txt, content-dev.txt, title-dev.txt
The format of corpus is as below:

content-train.txt
```shell
世间 本 没有 歧视 TAG_NAME_EN 歧视 源自于 人 的 内心 活动 TAG_NAME_EN “ 以爱 之 名 ” TAG_DATE 中国 艾滋病 反歧视 主题 创意 大赛 开幕 TAG_NAME_EN 让 “ 爱 ” 在 高校 流动 。 TAG_NAME_EN 详细 TAG_NAME_EN 
济慈 之 家 小朋友 感受 爱心 椅子  TAG_DATE TAG_NAME_EN 思源 焦点 公益 基金 向 盲童 孤儿院 “ 济慈 之 家 ” 提供 了 首 笔 物资 捐赠 。 这 笔 价值 近 万 元 的 物资 为 曲 美 家具 向 思源 · 焦点 公益 基金 提供 的 儿童 休闲椅 TAG_NAME_EN 将 用于 济慈 之 家 的 小孩子们 日常 使用 。 
...
```

title-train.txt
```shell
艾滋病 反歧视 创意 大赛 
思源 焦点 公益 基金 联手 曲 美 家具 共 献 爱心 ...
```

### Run the script
```shell
python headline.py
```

Reference
=============
* Tensorflow textsum examples on English Gigaword Corpus
https://github.com/tensorflow/models/tree/master/textsum
* Tensorflow seq2seq ops: 
https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
