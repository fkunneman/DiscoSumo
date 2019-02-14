
import os
import json
import numpy as np
import warnings
from collections import defaultdict
import itertools
warnings.filterwarnings("ignore")

from gensim.summarization import bm25
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity

class QuestionRelator:

    def __init__(self,w2v_model_path,tfidf_model_path,stopwords,w2v_dim=300):
        # initialize w2v
        print('Initializing word2vec')
        self.word2vec = Word2Vec.load(w2v_model_path)
        self.w2v_dim = w2v_dim
        
        # initialize softcosine
        print('Initializing softcosine')
        self.tfidf = TfidfModel.load(tfidf_model_path)
        with open(stopwords,'r',encoding='utf-8') as file_in:
            self.stopwords = file_in.read().strip().split('\n')
    
    def load_corpus(self,all_questions, labeled_questions, dictpath, topic_threshold = 0.1):
        with open(all_questions,'r',encoding='utf-8') as file_in:
            questions = json.loads(file_in.read())
        with open(labeled_questions,'r',encoding='utf-8') as file_in:
            labeled_questions = json.loads(file_in.read())
        labeled_ids = list(labeled_questions['test'].keys()) + list(labeled_questions['train'].keys())
        unlabeled_questions = [question for question in questions if not question['id'] in labeled_ids]
        labeled_questions = [question for question in questions if question['id'] in labeled_ids]

        self.corpus = [question['tokens'] for question in unlabeled_questions]
        if not os.path.exists(dictpath):
            self.dict = Dictionary(self.corpus)  # fit dictionary
            self.dict.save(dictpath)
        else:
            self.dict = Dictionary.load(dictpath)

        # save questions
        print('Saving questions')
        self.qid_question = {}
        self.unlabeled_questions = []
        self.labeled_questions = []
        for question in unlabeled_questions:
            qo = Question(question)
            self.unlabeled_questions.append(qo)
            self.qid_question[qo.qid] = qo
            
        # set bm25 model
        print('Initializing bm25 model')
        self.bm25_model = bm25.BM25([self.dict.doc2bow(q.tokens) for q in self.unlabeled_questions])
        
        # get average idf
        print('Calculating average idf')
        self.average_idf = sum(map(lambda k: float(self.bm25_model.idf[k]), self.bm25_model.idf.keys())) / len(self.bm25_model.idf.keys())         

    def encode_tokens(self,tokens):
        emb = []
        for w in tokens:
            try:
                emb.append(self.word2vec[w.lower()])
            except:
                emb.append(self.w2v_dim * [0])
        return emb
                    
    def softcosine(self, q1, q1emb, q2, q2emb):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = np.sqrt(dot(q1tfidf, q1emb, q1tfidf, q1emb))
        q2q2 = np.sqrt(dot(q2tfidf, q2emb, q2tfidf, q2emb))
        
        softcosine = dot(q1tfidf, q1emb, q2tfidf, q2emb) / (q1q1 * q2q2)
        return softcosine

    def softcosine_topics(self, t1, q2, q2emb):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        t1tfidf = self.tfidf[self.dict.doc2bow(t1)]
        t1emb = self.encode_tokens(t1)
        t1t1 = np.sqrt(dot(t1tfidf, t1emb, t1tfidf, t1emb))

        softcosines = []
        for i,t2 in enumerate(q2):
            t2tfidf = self.tfidf[self.dict.doc2bow([t2])]            
            t2t2 = np.sqrt(dot(t2tfidf, [q2emb[i]], t2tfidf, [q2emb[i]]))
            softcosines.append(dot(t1tfidf, t1emb, t2tfidf, [q2emb[i]]) / (t1t1 * t2t2))
            
        softcosine = max(softcosines)
        return softcosine

    def return_prominent_topics(self,topics,percentage=0.80):
        total = sum([x['topic_score'] for x in topics])
        acc = 0
        selection = []
        for topic in topics:
            selection.append(topic)
            acc += topic['topic_score']
            if (acc/total > percentage):
                break
        return selection

    def deduplicate_bm25_output(self,q_index,bm25_output):
        qids = []
        deduplicated = []
        for retrieved in bm25_output:
            to = self.unlabeled_questions[retrieved[0]]
            if to.qid != q_index:
                if to.qid not in qids:
                    qids.append(to.qid)
                    deduplicated.append(retrieved)
        return deduplicated
                    
    def select_candidates_bm25(self,question,topics,ntargets):
        scores_all = self.bm25_model.get_scores(self.dict.doc2bow(question.tokens))
        scores_numbers = [[i,score] for i,score in enumerate(scores_all)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        scores = scores_numbers_ranked[:ntargets]
        for topic in topics:
            scores_all = self.bm25_model.get_scores(self.dict.doc2bow(list(set(question.tokens) - set(topic['topic_text'].split()))))
            scores_numbers = [[i,score] for i,score in enumerate(scores_all)]
            scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
            scores.extend(scores_numbers_ranked[:ntargets])
        scores_ranked = sorted(scores,key = lambda k : k[1],reverse=True)
        scores_deduplicated = self.deduplicate_bm25_output(question.qid,scores_ranked)
        return scores_deduplicated
    
    def rank_questions_topics(self,question,targets,topics):
        scores_topic = []
        
        # score topics
        for target in targets:
            to = self.unlabeled_questions[target[0]]
            to.emb = self.encode_tokens(to.tokens)
            obj = [target[0],to]
            topic_sims = []
            for topic in topics:
                tokens = topic['topic_text'].split()
                sim = self.softcosine_topics(tokens, to.tokens, to.emb)
                topic_sim = topic['topic_score']*sim
                topic_sims.append(topic_sim)
                obj.extend([topic['topic_score'],sim,topic_sim])
            obj.append(np.mean(topic_sims))
            scores_topic.append(obj)
                        
        # rank scores
        scores_topic_ranked = sorted(scores_topic,key = lambda k : k[-1],reverse=True)

        return scores_topic_ranked

    def filter_questions(self,question,targets,filter_threshold):
        # filter by softcosine
        filtered = []
        disposed = []
        for target in targets:
            softcosine = self.softcosine(question.tokens,question.emb,target[1].tokens,target[1].emb)
            if softcosine < filter_threshold:
                target.append(softcosine)
                filtered.append(target)
            else:
                disposed.append(target)
        return filtered, disposed

    def rank_popularity(self,targets):
        targets_pop = []
        # add popularity info
        for target in targets:
            target.append(target[1].starcount)
            targets_pop.append(target)

        # rank by popularity
        indices_pop_ranked = [target[0] for target in sorted(targets_pop,key = lambda k : k[-1],reverse=True)]

        # combine rankings
        targets_combined = []
        for i,target in enumerate(targets_pop):
            target.extend([i,indices_pop_ranked.index(target[0])])
            target.append(np.mean([target[-2],target[-1]]))
            targets_combined.append(target)
        targets_combined_ranked = sorted(targets_combined,key = lambda k : k[-1])

        return targets_combined_ranked
        
    def diversify(self,targets,diversity_threshold):
        diversified = []
        remain = []
        for target in targets:
            diverse = True
            for target2 in diversified:
                softcosine = self.softcosine(target2[1].tokens,target2[1].emb,target[1].tokens,target[1].emb)
                if softcosine > diversity_threshold:
                    diverse = False
            if diverse:
                diversified.append(target)
            else:
                remain.append(target)
        return diversified, remain
    
    def relate_question(self,question,topic_percentage,ncandidates,filter_threshold,diversity_threshold):

        question.emb = self.encode_tokens(question.tokens)
        
        # select prominent topics
        prominent_topics = self.return_prominent_topics(question.topics,topic_percentage)
        print(len(prominent_topics),'topics')

        # retrieve candidate_questions
        candidates = self.select_candidates_bm25(question,prominent_topics,ncandidates)
        print(len(candidates),'candidates')
        
        # score and rank questions by topic
        candidates_ranked = self.rank_questions_topics(question,candidates,prominent_topics)
        print(len(candidates_ranked),'candidates_ranked')

        # filter questions
        candidates_ranked_filtered, candidates_disposed = self.filter_questions(question,candidates_ranked,filter_threshold)
        print(len(candidates_ranked_filtered),'filtered candidates')
        
        # add popularity information
        candidates_ranked_filtered_pop = self.rank_popularity(candidates_ranked_filtered)
        print(len(candidates_ranked_filtered_pop),'popular candidates')
        
        # diversify
        candidates_ranked_filtered_pop_diversified, candidates_ranked_filtered_pop_remain = self.diversify(candidates_ranked_filtered_pop,diversity_threshold)
        
        return prominent_topics, candidates, candidates_ranked, candidates_ranked_filtered, candidates_disposed, candidates_ranked_filtered_pop, candidates_ranked_filtered_pop_diversified, candidates_ranked_filtered_pop_remain
    

    def evaluate(self,pairs,labels):
        pass

class Question:
    
    def __init__(self,question):
        self.qid = question['id']
        self.cid = question['cid']
        self.question = question['questiontext']
        self.popularity = int(question['popularity'])
        self.answercount = int(question['answercount'])
        self.starcount = int(question['starcount'])
        self.topics = question['topics']
        self.tokens = [w.lower() for w in question['tokens']]
                 
    def set_emb(self,topic_emb,nontopic_emb,emb):
        self.topic_emb = topic_emb
        self.nontopic_emb = nontopic_emb
        self.emb = emb

    def set_tokens(self,topic_words,nontopic_words,all_words):
        self.topic_tokens = topic_words
        self.nontopic_tokens = nontopic_words
        self.all_tokens = all_words
