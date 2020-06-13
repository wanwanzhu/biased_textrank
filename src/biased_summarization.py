import nltk
import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('bert-base-nli-mean-tokens')


def cosine(u, v):
    return abs(1 - distance.cosine(u, v))


def rescale(a):
    maximum = np.max(a)
    minimum = np.min(a)
    return (a - minimum) / (maximum - minimum)


def biased_textrank(texts_embeddings, bias_embedding, damping_factor=0.8, similarity_threshold=0.8):
    # create text rank matrix, add edges between pieces that are more than X similar
    matrix = np.zeros((len(texts_embeddings), len(texts_embeddings)))
    for i, i_embedding in enumerate(texts_embeddings):
        for j, j_embedding in enumerate(texts_embeddings):
            if i == j:
                continue
            distance = cosine(i_embedding, j_embedding)
            if distance > similarity_threshold:
                matrix[i][j] = distance

    bias_weights = np.array([cosine(bias_embedding, embedding) for embedding in texts_embeddings])
    bias_weights = rescale(bias_weights)
    scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights

    for row in scaled_matrix:
        row /= np.sum(row)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    iterations = 40
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks


def get_sbert_embedding(text):
    if isinstance(text, list) or isinstance(text, tuple):
        return sbert.encode(text)
    else:
        return sbert.encode([text])


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s and not s.isspace()]
    return sentences


def load_text_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data


def main():
    left_data =


if __name__ == '__main__':
    main()
