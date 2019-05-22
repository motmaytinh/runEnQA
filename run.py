from preproc import word_tokenize
from preproc import convert_idx
from collections import Counter
from matplotlib import cm
import os
import numpy as np
import torch
import json
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

def predict(config, c, q):
    c_tokens = word_tokenize(c)
    q_tokens = word_tokenize(q)

    setattr(config, 'c_tokens', c_tokens)
    setattr(config, 'q_tokens', q_tokens)

    c_chars = [list(token) for token in c_tokens]
    q_chars = [list(token) for token in q_tokens]

    spans = convert_idx(c, c_tokens)

    c_limit = 400
    q_limit = 50
    char_limit = 16

    word_counter, char_counter = Counter(), Counter()

    for token in c_tokens:
        word_counter[token] += 1
        for char in token:
            char_counter[char] += 1

    with open('data/word2idx.json', 'r') as f:
        word2idx_dict = json.load(f)
    with open('data/char2idx.json', 'r') as f:
        char2idx_dict = json.load(f)

    c_word_idxs = np.zeros([c_limit], dtype=np.int32)
    c_char_idxs = np.zeros([c_limit, char_limit], dtype=np.int32)
    q_word_idxs = np.zeros([q_limit], dtype=np.int32)
    q_char_idxs = np.zeros([q_limit, char_limit], dtype=np.int32)

    def _get_word_idx(word):
        for w in (word, word.lower(), word.capitalize(), word.upper()):
            if w in word2idx_dict:
                return word2idx_dict[w]
            return 1

    def _get_char_idx(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(c_tokens):
        c_word_idxs[i] = _get_word_idx(token)


    for i, token in enumerate(c_chars):
        for j, char in enumerate(token):
            if j > char_limit:
                break
            c_char_idxs[i, j] = _get_char_idx(char)

    for i, token in enumerate(q_tokens):
        q_word_idxs[i] = _get_word_idx(token)


    for i, token in enumerate(q_chars):
        for j, char in enumerate(token):
            if j > char_limit:
                break
            q_char_idxs[i, j] = _get_char_idx(char)

    path_to_model = os.path.join("model", "model.pt")
    model = torch.load(path_to_model, map_location='cpu')

    model.eval()

    c_word_idxs = [c_word_idxs]
    c_char_idxs = [c_char_idxs]
    q_word_idxs = [q_word_idxs]
    q_char_idxs = [q_char_idxs]
    c_word_idxs = torch.tensor(c_word_idxs).long()
    c_char_idxs = torch.tensor(c_char_idxs).long()
    q_word_idxs = torch.tensor(q_word_idxs).long()
    q_char_idxs = torch.tensor(q_char_idxs).long()

    p1, p2 = model(config, c_word_idxs, c_char_idxs, q_word_idxs, q_char_idxs)

    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=1)

    outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
    for j in range(outer.size()[0]):
        outer[j] = torch.triu(outer[j])

    a1, _ = torch.max(outer, dim=2)
    a2, _ = torch.max(outer, dim=1)
    ymin = torch.argmax(a1, dim=1)
    ymax = torch.argmax(a2, dim=1)

    start_idx = spans[ymin][0]
    end_idx = spans[ymax][1]

    answer = c[start_idx:end_idx]
    return answer

def gen_heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, cmap=cm.Blues)

    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode=None)

    return im, cbar

if __name__ == '__main__':
    c = 'My name is Sang. I am twenty years old.'
    q = 'How old is Sang?'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    config = parser.parse_args()

    answer = predict(config, c, q)

    S = config.similarity_matrix
    S = S[0].data.numpy().transpose()
    c_tokens = config.c_tokens
    q_tokens = config.q_tokens

    S = S[:len(q_tokens), :len(c_tokens)]

    fig, ax = plt.subplots()

    im, cbar = gen_heatmap(S, q_tokens, c_tokens)

    fig.tight_layout()
    plt.show()

    print('Context: ', c)
    print('Question: ', q)
    print('Answer: ', answer)
