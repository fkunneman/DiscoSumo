__author__='thiagocastroferreira'

import json
import os
os.system('wget https://s3.eu-west-3.amazonaws.com/elasticbeanstalk-eu-west-3-026523518307/data.zip')
os.system('unzip data.zip')
os.remove('data.zip')

from main import GoeieVraag
from flask import Flask, request

application = Flask(__name__)

model = GoeieVraag()

@application.route("/rank", methods=['GET'])
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
        questions = { 'code':200, 'result': questions, 'method':method }

    return json.dumps(questions)

if __name__ == '__main__':
    application.debug = False
    application.run()