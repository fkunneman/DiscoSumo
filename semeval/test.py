import nltk

if __name__ == '__main__':
    question = 'Longest common subsequence long'
    for gram in nltk.ngrams(question.split(), 2):
        print(gram)

    for i in range(2, 4):
        q1 = ''
        for gram in nltk.ngrams(question.split(), i):
            q1 += '-'.join(gram) + ' '
        print(q1)