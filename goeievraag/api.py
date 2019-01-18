import json
import re

from flask import Flask, request
from main import GoeieVraag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets/'

model = GoeieVraag()

@app.route("/search/<query>", methods=['GET'])
def search(query=''):
    '''
    :return: return the 10 most semantic-similar questions to the query based on our official sysmte
    '''
    questions = {'code': 400}

    if request.method == 'GET' and query.strip() != '':
        questions = model(query.strip())
        questions = { 'code':200, 'result': [q[1] for q in questions] }

    return json.dumps(questions)

@app.route("/rank/<query>", methods=['GET'])
def rank(query):
    '''
    return the 10 most semantic-similar questions to the query based on our baseline (bm25)
    :param query:
    :return:
    '''
    questions = {'code': 400}

    if request.method == 'GET' and query.strip() != '':
        questions = model.retrieve(query.strip(), n=10)
        questions = { 'code':200, 'result': [q[1] for q in questions] }

    return json.dumps(questions)