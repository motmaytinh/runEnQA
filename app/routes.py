from flask import jsonify
from flask import request
from app import app
from run import predict
from run import gen_heatmap
import json
from argparse import Namespace
import matplotlib.pyplot as plt
import base64
import io


# @app.route('/')
@app.route('/en', methods=['POST'])
def index():
	caq = request.get_json()
	print(caq)
	context = caq['c']
	question = caq['q']
	# context = "I'm Quy. I'm 21 years old"
	# question = "How old is Quy"
	config = Namespace(batch_size=16)
	answer = predict(config, context, question)

	S = config.similarity_matrix
	S = S[0].data.numpy().transpose()
	c_tokens = config.c_tokens
	q_tokens = config.q_tokens

	S = S[:len(q_tokens), :len(c_tokens)]

	fig, ax = plt.subplots()

	im, cbar = gen_heatmap(S, q_tokens, c_tokens)

	fig.tight_layout()
	buf = io.BytesIO()
	plt.gcf().savefig(buf, format='png')
	buf.seek(0)
	plot = base64.b64encode(buf.read())

	return jsonify({'a': answer, 'p': str(plot)}) 
