import json

from flask import Flask, request
from main import GoeieVraag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets/'

model = GoeieVraag()

@app.route("/search/<query>", methods=['GET'])
def search(query=''):
    '''
    :return: return the 10 most likely questions for the query
    '''
    questions = {'code': 400}

    if request.method == 'GET' and query.strip() != '':
        questions = model(query)
        questions = { 'code':200, 'result': [q[1] for q in questions] }


    return json.dumps(questions)