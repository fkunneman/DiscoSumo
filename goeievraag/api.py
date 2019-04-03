import json

from flask import Flask, request
from main import GoeieVraag

app = Flask(__name__)

model = GoeieVraag()

@app.route("/rank/<query>", methods=['GET'])
@app.route("/rank/<query>/<method>", methods=['GET'])
def search(query='', method='ensemble'):
    '''
    :return: return the 10 most semantic-similar questions to the query based on our official sysmte
    '''
    questions = {'code': 400}

    if request.method == 'GET' and query.strip() != '':
        questions = model(query=query.strip(), method=method)
        questions = { 'code':200, 'result': questions }

    return json.dumps(questions)