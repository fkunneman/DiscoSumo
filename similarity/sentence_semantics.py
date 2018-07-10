
from gensim.models import Word2Vec
import gensim
import numpy
from scipy import spatial
import json
import itertools

class SentenceSemantics:

    def __init__(self):
        self.model = False
        self.lm = False
        self.index2word_se = False
        
    def train_model(self,sentences,size=100):
        self.model = Word2Vec(sentences,min_count=5,size=size,iter=1,workers=8)

    def save_model(self,outfile):
        self.model.wv.save_word2vec_format(outfile,binary=True)

    def load_model(self,modelfile):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(modelfile,binary=True, unicode_errors='ignore')
        self.index2word_set = set(self.model.wv.index2word)
        
    def load_lm(self,lm):
        # read in target
        with open(lm, 'r', encoding = 'utf-8') as file_in:
            model = json.load(file_in)
        self.lm = model[2]

    def remove_stopwords_sentence(self,sentence,stopwords):
        clean_sentence = list(set(sentence) - stopwords)
        return clean_sentence

    def remove_stopwords_sentences(self,sentences,stopwords):
        clean_sentences = []
        for sentence in sentences:
            clean_sentence = self.remove_stopwords_sentence(sentence,stopwords)
            clean_sentences.append(clean_sentence)
        return clean_sentences

    def avg_feature_vector(self, sentence, num_features):
        words = sentence.split()
        feature_vec = numpy.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in self.index2word_set:
                n_words += 1
                feature_vec = numpy.add(feature_vec, self.model[word])
        if (n_words > 0):
            feature_vec = numpy.divide(feature_vec, n_words)
        return feature_vec

    def sentence_similarity(self,sentence1,sentence2):
        s1_afv = self.avg_feature_vector(sentence1.lower(), num_features=100)
        s2_afv = self.avg_feature_vector(sentence2.lower(), num_features=100)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        return sim

    def find_related_word(self,sentence):
        sentence_cleaned = [w for w in sentence if w in self.index2word_set]
        try:
            return self.model.wv.most_similar(positive=sentence_cleaned)
        except:
            return [('-',0)]

    def word_similarity(self,word1,word2):
        if word1 in self.index2word_set and word2 in self.index2word_set:
            return self.model.wv.similarity(word1, word2)
        else:
            return 0
    
    def sentence_similarity_best_word(self,sentence1,sentence2):
        max_sim = 0
        best_sim = []
        for combi in itertools.product(sentence1,sentence2):
            sim = self.word_similarity(combi[0],combi[1])
            if sim > max_sim:
                max_sim = sim
                best_sim = [list(combi)]
            elif sim == max_sim:
                best_sim.append(combi)
        return max_sim, best_sim
    
    def rank_sentences_similarity(self,target_sentence,source_sentences):
        sentence_similarity = []
        for source_sentence in source_sentences:
            sensim = self.sentence_similarity(target_sentence,source_sentence)
            sentence_similarity.append((source_sentence,sensim))
        sentence_similarity_sorted = sorted(sentence_similarity,key=lambda k : k[1],reverse=True)
        return sentence_similarity_sorted

    def return_sentence_salient_word(self,sentence):
        word1_salience = [sentence[0],self.lm[sentence[0]]['count']]
        word_salience = [[word,self.lm[word]['count']] for word in sentence]
        return list(sorted(word_salience,key = lambda k : k[1]))[0] 

    def return_sentence_candidates(self,target_sentence,source_sentences,n_candidates):
        sorted_sentences = self.rank_sentences_similarity(target_sentence,source_sentences)
        possible_ranks = [1,200,1000,5000,2,100,200,10,50,2000,500,20,9999]
        candidate_ranks = possible_ranks[:n_candidates]
        candidate_ranks = [r-1 for r in candidate_ranks if r <= len(sorted_sentences)]
        sorted_sentences = self.rank_sentences_similarity(target_sentence,source_sentences)
        return [[self.return_sentence_salient_word(sorted_sentences[i][0]),sorted_sentences[i][0]] for i in candidate_ranks]
