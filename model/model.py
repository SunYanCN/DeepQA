from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from preprocess import one_hot, loaddata, tokenizer, pruning
from preprocess import punctuation_all, remove_stopwords, read_pre_word2vec

from numpy import zeros
import pickle
import os

class DataPath:
    def __init__(self):
        self.train_data_path = 'data/me_train.json'
        self.test_data_path = 'data/me_test.ann.json'
        self.validation_data_path = 'data/me_validation.ann.json'


def print_information(data, vocab_size, max_length, simple_len):
    print('Start printing ' + data + 'dataset information......')
    print('Number of ' + data + 'simples:', simple_len)
    print(data + 'vocab size:', vocab_size, 'unique words')
    print('documents max length:', max_length, 'words')


def text_preprocess(type='train'):
    # Read simples
    data_path = DataPath()
    train_simples = loaddata(data_path.train_data_path,data_type=type)
    # test_simples = loaddata(data_path.test_data_path,data_type=type)
    # validation_simples = loaddata(data_path.validation_data_path,data_type=type)

    # Remove punctuation
    questions, answers, documents = punctuation_all(train_simples[0:2])

    # Token text
    cut_quesions, cut_answers, cut_documents = tokenizer(questions, answers, documents)

    # Remove stopwords
    questions, answers, documents = remove_stopwords(cut_quesions, cut_answers, cut_documents)

    # Pruning sentences
    result = pruning(questions, answers, documents,plot_enable=True)


    # Padding
    vocab_size, encoded_docs, max_length, de_token_list = one_hot(train_simples)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print_information(type, vocab_size, max_length, simple_len)
    info_dict = {'vocab_size':vocab_size,'encoded_docs':encoded_docs,'max_length':max_length,'de_token_list':de_token_list,'padded_docs':padded_docs}

    # Serialization result
    with open('result.data') as f:
        pickle.dumps(result, f)

    with open('result_info.data') as f:
        pickle.dumps(info_dict,f)


# define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def pre_embedding_matrix(vocab_size, de_token_list, pre_words_embeddings):
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in de_token_list:
        embedding_vector = pre_words_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def bulid_model(vocab_size, max_length, padded_docs):
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size + 1, 64, weights=[], input_length=max_length,
                        trainable=False))  # keras进行embedding的时候必须进行len(vocab)+1
    model.add(LSTM(128, input_shape=(64, max_length), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0, batch_size=5)
    print('Accuracy: %f' % (accuracy * 100))


if __name__ == '__main__':

    if os.path.exists('result.data'):
        result = pickle.load('result.data')
    else:
        result = text_preprocess(type='train')

    pre_word2vec = read_pre_word2vec()
    embedding_matrix = pre_embedding_matrix()
    bulid_model()
