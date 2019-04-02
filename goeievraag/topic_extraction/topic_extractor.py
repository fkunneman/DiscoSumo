
import copy
import pickle
import numpy

from dte.functions import entity_extractor
from dte.classes import commonness

class TopicExtractor:

    def __init__(self,commonness_dir,ngram_entropy):
        print('Initializing commonness')
        self.set_commonness(commonness_dir)
        print('Initializing entropy')
        self.set_entropy(ngram_entropy)

    ############
    ### INIT ###
    ############

    def set_commonness(self,commonness_dir):
        self.cs = commonness.Commonness()
        self.cs.set_classencoder(commonness_dir + 'wiki_page.txt', commonness_dir + 'wiki_page.colibri.cls', commonness_dir + 'wiki_page.colibri.data')
        self.cs.set_dmodel(commonness_dir + 'all_ngrams.txt')
        
    def set_entropy(self,ngram_entropy):
        self.entropy = {}
        with open(ngram_entropy,'r',encoding='utf-8') as file_in:
            lines = file_in.read().strip().split('\n')
            for line in lines:
                tokens = line.split()
                entity = ' '.join(tokens[:-1])
                self.entropy[entity] = 1 - float(tokens[-1])
        self.entropy_set = set(self.entropy.keys())

    ###############
    ### HELPERS ###
    ###############

    def extract_sequence(self,question,wordtype):
        return sum([[word[wordtype].lower() for word in sentence] for sentence in question],[])

    def match_index(self,token,sequence1,sequence2):
        index = sequence1.index(token)
        return sequence2[index]

    def extract_ngrams(self,seq,n):
        ngrams = list(zip(*[seq[i:] for i in range(n)]))
        ngrams_string = [' '.join(ngram).lower() for ngram in ngrams]
        return ngrams_string

    def set_ceiling(self,entities,ceilingvalue):
        bounded_entities = []
        for entity in entities:
            entity = list(entity)
            if entity[1] > ceilingvalue:
                entity[1] = ceilingvalue
            bounded_entities.append(entity)
        return bounded_entities
    
    def filter_entities(self,entities,question):
        lemma_sequence = self.extract_sequence(question,'lemma')
        pos_sequence = self.extract_sequence(question,'pos')
        filtered = []
        for entity in entities:
            tokens = entity.split()
            if len(tokens) > 1:
                filtered.append(entity)
            else:
                entity = tokens[0]
                if len(entity) <= 1 or entity.isdigit():
                    continue
                try:
                    pos = self.match_index(entity,lemma_sequence,pos_sequence)
                    if not pos in ['lid','vnw','vz','bw','vg']:
                        filtered.append(entity)
                except:
                    print('COULD NOT FIND INDEX FOR',entity.encode('utf-8'),'in',' '.join(lemma_sequence).encode('utf-8'))
                    continue
        return filtered

    def rerank_topics(self,topics_commonness,topics_entropy):
        topics_commonness_txt = [x[0] for x in topics_commonness]
        topics_entropy_txt = [x[0] for x in topics_entropy]
        topics_commonness_only = list(set(topics_commonness_txt) - set(topics_entropy_txt))
        topics_entropy_only = list(set(topics_entropy_txt) - set(topics_commonness_txt))
        topics_union = list(set(topics_commonness_txt).union(set(topics_entropy_txt)))
        topics_commonness_complete = copy.deepcopy(topics_commonness)
        for topic in topics_entropy_only:
            topics_commonness_complete.append([topic,0])
        topics_entropy_complete = copy.deepcopy(topics_entropy)
        for topic in topics_commonness_only:
            topics_entropy_complete.append([topic,0])
        topics_commonness_complete_txt = [x[0] for x in topics_commonness_complete]
        topics_entropy_complete_txt = [x[0] for x in topics_entropy_complete]
        topics_combined = []
        for topic in topics_union:
            score_commonness = topics_commonness_complete[topics_commonness_complete_txt.index(topic)][1]
            score_entropy = topics_entropy_complete[topics_entropy_complete_txt.index(topic)][1]
            avg = numpy.mean([score_commonness,score_entropy])
            topics_combined.append([topic,avg,score_entropy,score_commonness])
        topics_ranked = sorted(topics_combined,key = lambda k : k[1],reverse=True)
        return topics_ranked

    def reduce_overlap(self,ranked_topics):
        filtered_topics = []
        for topic in ranked_topics:
            overlap = False
            for j,placed_topic in enumerate(filtered_topics):
                if set(topic[0].split()) & set(placed_topic[0].split()):
                    overlap = True
                    break
            if not overlap:
                filtered_topics.append(topic)
        return filtered_topics

    def topic2text(self,topics,question):
        lemma_sequence = self.extract_sequence(question,'lemma')
        text_sequence = self.extract_sequence(question,'text')
        topics_text = []
        for topic in topics:
            tokens = topic.split()
            if len(tokens) > 1:
                startindices = []
                for i in range(len(tokens)):
                    indices = [j for j,x in enumerate(lemma_sequence) if x == tokens[i]]
                    if len(startindices) == 0:
                        startindices = indices
                    else:
                        new_startindices = []
                        for index in indices:
                            for si in startindices:
                                if index == si+1:
                                    new_startindices.append(index)
                        if len(new_startindices) == 0:
                            print("could not find indices for",topic,lemma_sequence)
                        startindices = new_startindices
                # if not len(startindices) == 1:
                #     print("No single start index for",topic.encode('utf-8'),' '.join(lemma_sequence).encode('utf-8'),"startindex",startindices)
                index = startindices[0] - len(tokens)
                text = ' '.join(text_sequence[startindices[0]-1:startindices[0]+(len(tokens)-1)])
            else:
                entity = tokens[0]
                text = text_sequence[lemma_sequence.index(entity)]
            topics_text.append(text)
        return topics_text
    
    ###############
    ### EXTRACT ###
    ###############
    
    def extract_commonness(self,question):
        ee = entity_extractor.EntityExtractor()
        ee.set_commonness(self.cs)
        ee.extract_entities(self.extract_sequence(question,'lemma'))
        ee.filter_entities_threshold(0.01)
        entities = self.set_ceiling(ee.entities,1.0)
        entity_value = dict(entities)
        filtered_entities = self.filter_entities(entity_value.keys(),question)
        topics_commonness = [[e,entity_value[e]] for e in filtered_entities]
        return topics_commonness
                
    def extract_entropy(self,question):
        lemma_ngrams = []
        for sentence in question:
            lemma_ngrams.extend([self.extract_ngrams([word['lemma'] for word in sentence],token_length) for token_length in range(1,6)])
        lemma_ngrams = list(set(sum(lemma_ngrams,[])) & self.entropy_set)
        filtered_entities = self.filter_entities(lemma_ngrams,question)
        topics_entropy = [[e,self.entropy[e]] for e in filtered_entities] 
        return topics_entropy
    
    def extract(self,question,max_topics=5):
        topics_commonness = self.extract_commonness(question)
        topics_entropy = self.extract_entropy(question)
        topics_ranked = self.rerank_topics(topics_commonness,topics_entropy)
        topics_filtered = self.reduce_overlap(topics_ranked)[:max_topics]
        topics_text = self.topic2text([x[0] for x in topics_filtered],question)
        topics_filtered_text = [tf + [topics_text[i]] for i,tf in enumerate(topics_filtered)]
        topics_filtered_text_dict = [{'topic':x[0],'topic_score':x[1],'topic_entropy':x[2],'topic_commonness':x[3],'topic_text':x[4]} for x in topics_filtered_text]
        return topics_filtered_text_dict

    def extract_list(self,questions,max_topics=5):
        return [self.extract(q,max_topics) for q in questions]
