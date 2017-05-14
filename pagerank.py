import file_reader as fr
import re
import numpy as np

WINDOW_SIZE = 10
RANDOM_JUMP_PROB = 0.15
EPSILON = 0.00001
K = 100  # how many of top results will be logged

# merging train and test texts
texts_pos = fr.read_files('test', 'pos') + fr.read_files('train', 'pos')
texts_neg = fr.read_files('test', 'neg') + fr.read_files('train', 'neg')


def construct_graph_and_run_pagerank_algorithm(texts):
    regex = r"[a-z]+['[a-z]+]?\b"  # stop words included
    words = []
    for text in texts:
        words += re.findall(regex, text)
    all_words = list(set(words))
    l = len(all_words)

    # needed to make the task for searching index of a word faster.
    hash_index = {}
    for i in range(l):
        hash_index[all_words[i]] = i

    matrix = np.zeros((l, l), dtype=np.float32)  # working with float32 to save space.

    # adds nodes to graph: constructing a symmetric matrix due to undirected property of our graph
    def add_to_matrix(word1, word2):
        m = hash_index[word1]
        n = hash_index[word2]
        matrix[m][n] += 1
        matrix[n][m] += 1

    # populate graph
    print('Constructing graph...')
    num = 0
    for text in texts:
        words = re.findall(regex, text)
        i = 0
        # connecting first WINDOW_SIZE words to each other
        for i in range(WINDOW_SIZE):
            for j in range(i + 1, WINDOW_SIZE):
                add_to_matrix(words[i], words[j])
        # connecting the rest of the words to each other with window size.
        for i in range(i + 1, len(words)):
            for j in range(i - WINDOW_SIZE, i):
                add_to_matrix(words[i], words[j])
        num += 1
        if num % int(len(texts)/10) == 0:
            print(str(int(100 * num / len(texts))) + '%')

    print('Constructing transition probability matrix with teleporting...')
    num = 0
    for i in range(l):
        matrix[i] /= sum(matrix[i])
        matrix[i] *= 1 - RANDOM_JUMP_PROB
        matrix[i] += RANDOM_JUMP_PROB / l
        num += 1
        if num % int((l/10)) == 0:
            print(str(int(100 * num / l)) + '%')

    print('Starting random walk...')
    # random start
    x = np.zeros(l, dtype=np.float32) + 1.0 / l
    norm = 1
    c = 0
    while norm > EPSILON:
        xP = x.dot(matrix)
        norm = np.sqrt(np.square(x-xP).sum())  # L2-norm
        x = xP
        c += 1
        if c == 1000:  # threshold
            break
        print(str(c) + '. power iteration... ' + str(norm))
    ranks = x

    top_salient_word_indices = ranks.argsort()[-K:][::-1]
    for i in range(K):
        print(str(i+1) + ' : ' + all_words[top_salient_word_indices[i]])

print('Positive Reviews')
construct_graph_and_run_pagerank_algorithm(texts_pos)
print('Negative Reviews')
construct_graph_and_run_pagerank_algorithm(texts_neg)
