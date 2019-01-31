
import pickle

class QCat:
    def __init__(self,model,le,cat2id,vocab):
        with open(model,'rb') as fid:
            self.model = pickle.load(fid)
        self.vocabulary = {}
        vocabulary_in = open(vocab,mode="r",encoding="utf-8")
        for i,line in enumerate(vocabulary_in.readlines()):
            self.vocabulary[line.strip().split('\t')[0]] = i
        vocabulary_in.close()
        self.vocab_size = len(self.vocabulary.keys())
        self.le = {}
        with open(le,'r',encoding='utf-8') as file_in:
            for line in file_in.read().strip().split('\n'):
                tokens = line.split()
                self.le[int(tokens[-1])] = ' '.join(tokens[:-1])
        self.cat2id = {}
        self.cat2parent = {}
        with open(cat2id,'r',encoding='utf-8') as file_in:
            for line in file_in.read().strip().split('\n'):
                tokens = line.split('\t')
                self.cat2id[tokens[0]] = tokens[1]
                if tokens[3] == '1' and tokens[1] != '15':
                    self.cat2parent[tokens[0]] = tokens[0]
                else:
                    self.cat2parent[tokens[0]] = tokens[2]

    def vectorize_question(self, question):
        vectorframe = [float(0)] * self.vocab_size
        tokens = question.lower().split()
        ngrams = tokens + [" ".join(x) for x in zip(tokens, tokens[1:])] + [" ".join(x) for x in zip(tokens, tokens[1:], tokens[2:])]
        vector = vectorframe[:]
        for feature in ngrams:
            try:
                vector[self.vocabulary[feature]] = float(1)
            except KeyError:
                continue
        return vector
    
    def return_top_cats(self,question,ncats=5):
        instance = self.vectorize_question(question)
        prediction_probs = self.model.predict_proba([instance])
        # sort probs
        prediction_probs_sorted = sorted([[i,x] for i,x in enumerate(prediction_probs[0])],key = lambda k : k[1],reverse=True)
        # convert to labels
        predictions = []
        for i in range(ncats):
            predictions.append([self.le[prediction_probs_sorted[i][0]],self.cat2id[self.le[prediction_probs_sorted[i][0]]],prediction_probs_sorted[i][1]])
        return predictions

    def main(self,questions,ncats):
        results = []
        for question in questions:
            results.append(self.return_top_cats(question,ncats))
        return results

    def __call__(self, question, ncats=5):
        return self.return_top_cats(question, ncats)
