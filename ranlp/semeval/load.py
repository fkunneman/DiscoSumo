__author__='thiagocastroferreira'

import os
import paths
import xml.etree.ElementTree as ET

DATASET_PATH=paths.DATASET_PATH
TRAIN_PATH=os.path.join(DATASET_PATH, 'train', 'SemEval2016-Task3-CQA-QL-train-part1.xml')
TRAIN_PATH_PART2=os.path.join(DATASET_PATH, 'train', 'SemEval2016-Task3-CQA-QL-train-part2.xml')
DEV_PATH=os.path.join(DATASET_PATH, 'dev', 'SemEval2016-Task3-CQA-QL-dev.xml')
TEST2016_PATH=os.path.join(DATASET_PATH, 'test2016', 'English', 'SemEval2016-Task3-CQA-QL-test.xml')
TEST2017_PATH=os.path.join(DATASET_PATH, 'test2017', 'English', 'SemEval2017-task3-English-test.xml')

def load(path, set_='train'):
    tree = ET.parse(path)

    questions = {}
    questions_xml = tree.findall('OrgQuestion')

    for question_xml in questions_xml:
        # question information
        qid = question_xml.attrib['ORGQ_ID']
        if qid not in questions:
            question = {}
            question['id'] = question_xml.attrib['ORGQ_ID']
            question['subject'] = question_xml.find('OrgQSubject').text
            question['body'] = question_xml.find('OrgQBody').text
            question['set'] = set_

            question['duplicates'] = []
            questions[qid] = question
        else:
            question = questions[qid]

        thread_xml = question_xml.find('Thread')
        # Related question information
        rel_question_xml = thread_xml.find('RelQuestion')
        rel_question = {}
        rel_question['id'] = rel_question_xml.attrib['RELQ_ID']
        rel_question['ranking'] = rel_question_xml.attrib['RELQ_RANKING_ORDER']
        rel_question['category'] = rel_question_xml.attrib['RELQ_CATEGORY']
        rel_question['userid'] = rel_question_xml.attrib['RELQ_USERID']
        rel_question['username'] = rel_question_xml.attrib['RELQ_USERNAME']
        rel_question['relevance'] = rel_question_xml.attrib['RELQ_RELEVANCE2ORGQ']
        rel_question['subject'] = rel_question_xml.find('RelQSubject').text
        rel_question['body'] = rel_question_xml.find('RelQBody').text

        # related comments information
        rel_comments_xml = thread_xml.findall('RelComment')
        rel_comments = []
        for rel_comment_xml in rel_comments_xml:
            rel_comment = {}
            rel_comment['id'] = rel_comment_xml.attrib['RELC_ID']
            rel_comment['date'] = rel_comment_xml.attrib['RELC_DATE']
            rel_comment['userid'] = rel_comment_xml.attrib['RELC_USERID']
            rel_comment['username'] = rel_comment_xml.attrib['RELC_USERNAME']
            rel_comment['relevance2question'] = rel_comment_xml.attrib['RELC_RELEVANCE2ORGQ']
            rel_comment['relevance2relquestion'] = rel_comment_xml.attrib['RELC_RELEVANCE2RELQ']
            rel_comment['text'] = rel_comment_xml.find('RelCText').text
            rel_comments.append(rel_comment)

        duplicate = {
            'rel_question': rel_question,
            'rel_comments': rel_comments
        }
        question['duplicates'].append(duplicate)
    return questions

def run():
    trainset, devset = load(TRAIN_PATH, set_='train1'), load(DEV_PATH, set_='dev')
    trainset.update(load(TRAIN_PATH_PART2, set_='train2'))

    testset2016, testset2017 = load(TEST2016_PATH, set_='test2016'), load(TEST2017_PATH, set_='test2017')
    return trainset, devset, testset2016, testset2017

def rank(ranking):
    _ranking = []
    for i, q in enumerate(sorted(ranking, key=lambda x: x[1], reverse=True)):
        _ranking.append({'Answer_ID':q[2], 'SCORE':q[1], 'RANK':i+1, 'LABEL':q[0]})
    return _ranking

def save(ranking, fname):
    f = open(fname, 'w')
    query_ids = sorted(list(ranking.keys()))
    for query_id in query_ids:
        rel_question_ids = sorted(map(lambda x: x['Answer_ID'], ranking[query_id]), key=lambda x: int(x.split('_R')[1]))
        for rel_question_id in rel_question_ids:
            rel_question = list(filter(lambda x: x['Answer_ID'] == rel_question_id, ranking[query_id]))[0]
            f.write('\t'.join([query_id, rel_question['Answer_ID'], str(0), str(rel_question['SCORE']), rel_question['LABEL'], '\n']))
    f.close()
