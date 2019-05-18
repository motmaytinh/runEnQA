from flask import jsonify
from flask import request
from app import app
from run import predict
import json


# @app.route('/')
@app.route('/en', methods=['POST'])
def index():
	caq = request.get_json()
	print(caq)
	context = caq['c']
	question = caq['q']
	# context = "I'm Quy. I'm 21 years old"
	# question = "How old is Quy"
	answer = predict(context, question)
	return jsonify({'a': answer}) 
