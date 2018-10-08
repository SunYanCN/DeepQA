import numpy as np
import jieba
import ujson
import pandas as pd
from jieba import lcut
from tqdm import tqdm
from pprint import pprint
from collections import Counter
from tools import flatten

from tools import dereplication
from tools import punctuation_line
from gensim.models import KeyedVectors
import matplotlib.pylab as plt

# Ignore some warning messages
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Set the log print content
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
jieba.setLogLevel(logging.INFO)


# Read pre-training word2vector
def read_pre_word2vec():
    model = KeyedVectors.load_word2vec_format('F:/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.bz2')
    return model


# Load data file into memory
def loaddata(filename, data_type='train'):
    with open(filename, 'r', encoding='utf8') as file:
        train_data = ujson.loads(file.read())

    questions = []
    answers = []
    documents = []
    for _, value in train_data.items():
        question = value['question']
        for _, evidence in value['evidences'].items():
            questions.append(question)
            answer = ''.join(evidence['answer'])
            answers.append(answer)
            document = evidence['evidence']
            documents.append(document)

    train_simple = map(lambda x, y, z: [x, y, z], questions, answers, documents)

    train_df = pd.DataFrame([item for item in train_simple], columns=['question', 'answer', 'document'])
    print(train_df.describe())
    train_df.to_csv(data_type + '.csv', index=False, header=True)

    return [questions, answers, documents, train_simple]


# Word segmentation
def tokenizer(questions, answers, documents):
    cut_documents = map(lcut, documents)
    cut_answers = map(lcut, answers)
    cut_quesions = map(lcut, questions)
    return [cut_quesions, cut_answers, cut_documents]


# Remove punctuation
# Reference: https://zhon.readthedocs.io/en/latest
def punctuation_all(questions, answers, documents):
    questions = map(punctuation_line, questions)
    answers = map(lambda x: x, answers)
    documents = map(punctuation_line, documents)
    return [questions, answers, documents]


# Get a list of Chinese stop words
# Reference: https://github.com/goto456/stopwords?1536998864835
def get_stopwords():
    stopwords_file = 'HIT_stop_list.txt'
    stop_f = open(stopwords_file, "r", encoding='utf-8')
    stop_words = list()
    for line in stop_f.readlines():
        line = line.strip()
        if not len(line):
            continue
        stop_words.append(line)
    stop_f.close()
    print('哈工大停止词表长度为：' + str(len(stop_words)))
    return stop_words


def remove_stopwords():
    stop_words = get_stopwords()


# Create a dictionary of word and index conversions
def creat_lookup_table(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


# Batch data acquisition
def generate_batches(int_simple, batch_size, steps):
    simple_size = len(int_simple)
    n_batches = simple_size // (batch_size * steps)

    int_simple = int_simple[:n_batches * (batch_size * steps)]
    int_simple = int_simple.reshape((batch_size, -1))

    for n in range(0, int_simple.shape[1], steps):
        x = int_simple[:, n:n + steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


# Code conversion of words
def one_hot(token):
    token_list = flatten(token)
    de_token_list = dereplication(token_list)
    vocab_size = len(de_token_list)
    token_length = map(len, token)
    max_length = max(token_length)
    mean_length = np.mean([i for i in token_length])

    vocab_to_int, int_to_vocab = creat_lookup_table(de_token_list)

    encoded_docs = []
    for i in token:
        encoded_docs.append([vocab_to_int.get(j) for j in i])
    return vocab_size, encoded_docs, max_length, de_token_list, mean_length
    # print(encoded_docs)
    # print(vocab_to_int)


# Pruning sentences over a certain length
def pruning(questions, answers, documents,num_tokens, plot_enable=True):
    plt.hist(np.log(num_tokens), bins=100)
    plt.xlim((0, 10))
    plt.ylabel('number of tokens')
    plt.xlabel('length of tokens')
    plt.title('Distribution of tokens length')
    plt.show()

    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    print(max_tokens)

    percentage = np.sum(num_tokens < max_tokens) / len(num_tokens)
    print(percentage)


if __name__ == '__main__':
    words = ['123', '213', '321']
    data = np.arange(12)
    batches = generate_batches(data, 2, 1)
    print([i for i in batches])
    vocab_to_int, int_to_vocab = creat_lookup_table(words)
    print(vocab_to_int, int_to_vocab)
