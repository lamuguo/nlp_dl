# -*- coding: utf-8 -*-


from __future__ import print_function
from nltk.tokenize import word_tokenize
import jieba
import sys
import numpy as np
from nltk.stem import SnowballStemmer

sys.path.append('../utils/')
from rank_io import *

class Preprocess(object):

    _valid_lang = ['en', 'cn']
    _stemmer = SnowballStemmer('english')

    def __init__(self,
                 lang='en',
                 stop_words=list(),
                 min_freq=1,
                 max_freq=sys.maxint,
                 min_len=0,
                 max_len=sys.maxint,
                 word_dict=None,
                 words_useless=None):
        assert lang.lower() in Preprocess._valid_lang, 'Wrong language type: %s' % lang
        self._lang = lang
        self._stop_words = stop_words
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._min_len = min_len
        self._max_len = max_len
        self._word_dict = word_dict
        self._words_useless = words_useless
        self._words_df = dict()

    def run(self, file_path):
        dids, docs = Preprocess.load(file_path)
        docs = Preprocess.word_seg(docs, self._lang)
        dids, docs = Preprocess.doc_filter(dids, docs, self._min_len, self._max_len)
        docs = Preprocess.word_stem(docs)
        docs, self._words_useless = Preprocess.word_filter(docs,
                                                           words_useless=self._words_useless,
                                                           stop_words=self._stop_words,
                                                           min_freq=self._min_freq,
                                                           max_freq=self._max_freq)
        docs, self._word_dict = Preprocess.word_index(docs, word_dict=self._word_dict)
        return dids, docs

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = open(file_path, 'r')
        for line in f:
            line = line.decode('utf8')
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def word_seg_en(docs):
        docs = [word_tokenize(sent) for sent in docs]
        return docs

    @staticmethod
    def word_seg_cn(docs):
        docs = [list(jieba.cut(sent)) for sent in docs]
        return docs

    @staticmethod
    def word_seg(docs, lang):
        assert lang.lower() in Preprocess._valid_lang, 'Wrong language type: %s' % lang
        docs = getattr(Preprocess, '%s_%s' % (sys._getframe().f_code.co_name, lang))(docs)
        return docs

    @staticmethod
    def cal_doc_freq(docs):
        wdf = dict()
        for ws in docs:
            ws = set(ws)
            for w in ws:
                wdf[w] = wdf.get(w, 0) + 1
        return wdf

    @staticmethod
    def word_filter(docs,
                    words_useless=None,
                    stop_words=list(),
                    min_freq=1,
                    max_freq=sys.maxint):
        if words_useless is None:
            words_useless = set()
            # filter with stop_words
            words_useless.update(stop_words)
            # filter with min_freq and max_freq
            wdf = Preprocess.cal_doc_freq(docs)
            for w in wdf:
                if min_freq > wdf[w] or max_freq < wdf[w]:
                    words_useless.add(w)
        # filter with useless words
        docs = [[w for w in ws if w not in words_useless] for ws in docs]
        return docs, words_useless

    @staticmethod
    def doc_filter(dids, docs, min_len=1, max_len=sys.maxint):
        new_docs = list()
        new_dids = list()
        for i in range(len(docs)):
            if min_len <= len(docs[i]) <= max_len:
                new_docs.append(docs[i])
                new_dids.append(dids[i])
        return new_dids, new_docs

    @staticmethod
    def word_stem(docs):
        docs = [[Preprocess._stemmer.stem(w) for w in ws] for ws in docs]
        return docs

    @staticmethod
    def build_word_dict(docs):
        word_dict = dict()
        for ws in docs:
            for w in ws:
                word_dict.setdefault(w, len(word_dict))
        return word_dict

    @staticmethod
    def word_index(docs, word_dict=None):
        if word_dict is None:
            word_dict = Preprocess.build_word_dict(docs)
        docs = [[word_dict[w] for w in ws if w in word_dict] for ws in docs]
        return docs, word_dict

    @staticmethod
    def save_lines(file_path, lines):
        f = open(file_path, 'w')
        for line in lines:
            line = line.encode('utf8')
            f.write(line + "\n")
        f.close()

    @staticmethod
    def load_lines(file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return lines

    @staticmethod
    def save_dict(file_path, dic):
        lines = ['%s %s' % (k, v) for k, v in dic.iteritems()]
        Preprocess.save_lines(file_path, lines)

    @staticmethod
    def load_dict(file_path):
        lines = Preprocess.load_lines(file_path)
        dic = dict()
        for line in lines:
            k, v = line.split()
            dic[k] = v
        return dic

    def save_words_useless(self, words_useless_fp):
        Preprocess.save_lines(words_useless_fp, self._words_useless)

    def load_words_useless(self, words_useless_fp):
        self._words_useless = set(Preprocess.load_lines(words_useless_fp))

    def save_word_dict(self, word_dict_fp):
        Preprocess.save_dict(word_dict_fp, self._word_dict)

    def load_word_dict(self, word_dict_fp):
        self._word_dict = Preprocess.load_dict(word_dict_fp)

    def save_words_df(self, words_df_fp):
        Preprocess.save_dict(words_df_fp, self._words_df)

    def load_words_df(self, words_df_fp):
        self._words_df = Preprocess.load_dict(words_df_fp)


class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([words[i], words[i + k]]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in xrange(L - 2):
                for j in xrange(i + 1, L - 1):
                    for k in xrange(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
            Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                for j in xrange(i + 1, L - 2):
                    for k in xrange(j + 1, L - 1):
                        for l in xrange(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)

def cal_hist(t1_rep, t2_rep, qnum, hist_size):
    #qnum = len(t1_rep)
    mhist = np.zeros((qnum, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))
    for (i,j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) / 2. * (hist_size - 1.))
        mhist[i][vid] += 1.
    mhist += 1.
    mhist = np.log10(mhist)
    return mhist.flatten()

def _test_preprocess():
    file_path = '/Users/houjianpeng/tmp/txt'
    preprocessor = Preprocess()
    dids, docs = preprocessor.run(file_path)
    print(dids)
    print(docs)
    preprocessor.save_word_dict(file_path + '.word_dict')
    preprocessor.save_words_df(file_path + '.words_df')
    preprocessor.save_words_useless(file_path + '.words_useless')
    preprocessor.load_words_useless(file_path + '.words_useless')


def _test_ngram():
    words = 'hello, world! hello, deep!'
    print(NgramUtil.ngrams(list(words), 3, ''))

def _test_hist():
    embedfile = '../../data/mq2007/embed_wiki-pdc_d50_norm'
    queryfile = '../../data/mq2007/qid_query.txt'
    docfile = '../../data/mq2007/docid_doc.txt'
    relfile = '../../data/mq2007/relation.test.fold5.txt'
    histfile = '../../data/mq2007/relation.test.fold5.hist-30.txt'
    embed_dict = read_embedding(filename = embedfile)
    print('after read embedding ...')
    _PAD_ = 193367
    embed_dict[_PAD_] = np.zeros((50, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [193368, 50]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    query, _ = read_data(queryfile)
    print('after read query ....')
    doc, _ = read_data(docfile)
    print('after read doc ...')
    rel = read_relation(relfile)
    print('after read relation ... ')
    fout = open(histfile, 'w')
    for label, d1, d2 in rel:
        assert d1 in query
        assert d2 in doc
        qnum = len(query[d1])
        d1_embed = embed[query[d1]]
        d2_embed = embed[doc[d2]]
        curr_hist = cal_hist(d1_embed, d2_embed, qnum, 30)
        curr_hist = curr_hist.tolist()
        fout.write(' '.join(map(str, curr_hist)))
        fout.write('\n')
        print(qnum)
        #print(curr_hist)
    fout.close()



if __name__ == '__main__':
    #_test_ngram()
    #_test_hist()
    path = '/home/fanyixing/dataset/marco/'
    infile_path = path + 'did.test.txt'
    outfile_path = path + 'did.test.processed.txt'
    #infile_path = './did.train.txt'
    #outfile_path = 'did.train.processed.txt'
    dictfile_path = path + 'word_dict.txt'
    dffile_path = path + 'word_df.txt'
    preprocessor = Preprocess(min_freq = 5)
    preprocessor.load_word_dict(dictfile_path)
    preprocessor.load_words_df(dffile_path)
    dids, docs = preprocessor.run(infile_path)

    fout = open(outfile_path,'w')
    for inum,did in enumerate(dids):
        fout.write('%s\t%s\n'%(did, ' '.join(map(str,docs[inum]))))
    fout.close()
    print('Done ...')
