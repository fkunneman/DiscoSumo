import json

from flask import Flask, request
from main import GoeieVraag

app = Flask(__name__)

model = GoeieVraag()

@app.route("/rank", methods=['GET'])
def search():
    '''
    :return: return the 10 most semantic-similar questions to the query based on our official sysmte
    '''
    questions = {'code': 400}

    query, method = '', 'ensemble'
    if 'q' in request.args:
        query = request.args['q'].strip()
    if 'method' in request.args:
        method = request.args['method'].strip()

    if request.method == 'GET':
        questions = model(query=query.strip(), method=method)
        questions = { 'code':200, 'result': questions }

    return json.dumps(questions)