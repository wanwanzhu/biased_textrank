import json
from os import listdir
from os.path import isfile, join

import nltk
import numpy as np
from rouge import Rouge
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('bert-base-nli-mean-tokens')
rouge = Rouge()


def cosine(u, v):
    return abs(1 - distance.cosine(u, v))


def rescale(a):
    maximum = np.max(a)
    minimum = np.min(a)
    return (a - minimum) / (maximum - minimum)


def biased_textrank(texts_embeddings, bias_embedding, damping_factor=0.80, similarity_threshold=0.8, biased=True):
    # create text rank matrix, add edges between pieces that are more than X similar
    matrix = np.zeros((len(texts_embeddings), len(texts_embeddings)))
    for i, i_embedding in enumerate(texts_embeddings):
        for j, j_embedding in enumerate(texts_embeddings):
            if i == j:
                continue
            distance = cosine(i_embedding, j_embedding)
            if distance > similarity_threshold:
                matrix[i][j] = distance

    if biased:
        bias_weights = np.array([cosine(bias_embedding, embedding) for embedding in texts_embeddings])
        bias_weights = rescale(bias_weights)
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias_weights
    else:
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) / len(matrix)

    for row in scaled_matrix:
        row /= np.sum(row)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    iterations = 120
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


def get_filenames_in_directory(path):
    return [filename for filename in listdir(path) if isfile(join(path, filename))]


def select_top_k_texts_preserving_order(texts, ranking, k):
    texts_sorted = sorted(zip(texts, ranking), key=lambda item: item[1], reverse=True)
    top_texts = texts_sorted[:k]
    top_texts = [t[0] for t in top_texts]
    result = []
    for text in texts:
        if text in top_texts:
            result.append(text)
    return result


def main():
    democratic_bias = '''The Democratic Party's philosophy of modern liberalism advocates social and economic equality, along with the welfare state. It seeks to provide government regulation in the economy to promote the public interest. Environmental protection, support for organized labor, maintenance and expansion of social programs, affordable college tuition, universal health care, equal opportunity, and consumer protection form the core of the party's economic policy. On social issues, it advocates campaign finance reform, LGBT rights, criminal justice and immigration reform, stricter gun laws, and the legalization of marijuana.'''
    democratic_bias_embedding = get_sbert_embedding(democratic_bias)
    republican_bias = '''The 21st-century Republican Party ideology is American conservatism, which incorporates both economic policies and social values. The GOP supports lower taxes, free market capitalism, restrictions on immigration, increased military spending, gun rights, restrictions on abortion, deregulation and restrictions on labor unions. After the Supreme Court's 1973 decision in Roe v. Wade, the Republican Party opposed abortion in its party platform and grew its support among evangelicals. The GOP was strongly committed to protectionism and tariffs at its founding but grew more supportive of free trade in the 20th century.'''
    republican_bias_embedding = get_sbert_embedding(republican_bias)
    data_path = '../data/us-presidential-debates/'
    democrat_path = data_path + 'democrat/'
    republican_path = data_path + 'republican/'
    transcript_path = data_path + 'transcripts/'
    democrat_gold_standards = [{'filename': filename, 'content': load_text_file(democrat_path + filename)} for filename
                               in get_filenames_in_directory(democrat_path)]
    republican_gold_standards = [{'filename': filename, 'content': load_text_file(republican_path + filename)} for
                                 filename in get_filenames_in_directory(republican_path)]
    transcripts = [{'filename': filename, 'content': load_text_file(transcript_path + filename)}
                   for filename in get_filenames_in_directory(transcript_path)]
    democrat_summaries = [{'filename': filename} for filename in get_filenames_in_directory(democrat_path)]
    republican_summaries = [{'filename': filename} for filename in get_filenames_in_directory(republican_path)]
    normal_summaries = [{'filename': filename} for filename in get_filenames_in_directory(transcript_path)]
    for i, transcript in enumerate(transcripts):
        transcript_sentences = get_sentences(transcript['content'])
        transcript_sentence_embeddings = [get_sbert_embedding(sentence) for sentence in transcript_sentences]
        if len(democrat_gold_standards[i]['content']) != 0:
            democratic_ranks = biased_textrank(transcript_sentence_embeddings, democratic_bias_embedding)
            democrat_summary = ' '.join(select_top_k_texts_preserving_order(transcript_sentences, democratic_ranks, 20))
            democrat_summaries[i]['content'] = democrat_summary
        if len(republican_gold_standards[i]['content']) != 0:
            republican_ranks = biased_textrank(transcript_sentence_embeddings, republican_bias_embedding)
            republican_summary = ' '.join(select_top_k_texts_preserving_order(transcript_sentences, republican_ranks, 20))
            republican_summaries[i]['content'] = republican_summary
        if len(republican_gold_standards[i]['content']) != 0 or len(democrat_gold_standards[i]['content']) != 0:
            normal_ranks = biased_textrank(transcript_sentence_embeddings, None, damping_factor=0.85, biased=False)
            normal_summary = ' '.join(select_top_k_texts_preserving_order(transcript_sentences, normal_ranks, 20))
            normal_summaries[i]['content'] = normal_summary

    # saving results
    with open('democrat_summaries.json', 'w') as f:
        f.write(json.dumps(democrat_summaries))
    with open('republican_summaries.json', 'w') as f:
        f.write(json.dumps(republican_summaries))
    with open('normal_summaries.json', 'w') as f:
        f.write(json.dumps(normal_summaries))

    # load results
    # with open('democrat_summaries.json') as f:
    #     democrat_summaries = json.load(f)
    # with open('republican_summaries.json') as f:
    #     republican_summaries = json.load(f)

    # evaluation
    democrat_rouge_scores = calculate_rouge_score(democrat_gold_standards, democrat_summaries)
    print('Democrat Results:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(democrat_rouge_scores['rouge-1']),
                                                         np.mean(democrat_rouge_scores['rouge-2']),
                                                         np.mean(democrat_rouge_scores['rouge-l'])))
    print('############################')

    republican_rouge_scores = calculate_rouge_score(democrat_gold_standards, republican_summaries)
    print('Republican Results against Democrat Gold Standard:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(republican_rouge_scores['rouge-1']),
                                                         np.mean(republican_rouge_scores['rouge-2']),
                                                         np.mean(republican_rouge_scores['rouge-l'])))
    print('############################')

    normal_rouge_scores = calculate_rouge_score(democrat_gold_standards, normal_summaries)
    print('Normal Results against Democrat Gold Standard:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(normal_rouge_scores['rouge-1']),
                                                         np.mean(normal_rouge_scores['rouge-2']),
                                                         np.mean(normal_rouge_scores['rouge-l'])))
    print('############################')

    republican_rouge_scores = calculate_rouge_score(republican_gold_standards, republican_summaries)
    print('Republican Results:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(republican_rouge_scores['rouge-1']),
                                                         np.mean(republican_rouge_scores['rouge-2']),
                                                         np.mean(republican_rouge_scores['rouge-l'])))
    print('############################')

    democrat_rouge_scores = calculate_rouge_score(republican_gold_standards, democrat_summaries)
    print('Democrat Results against Republican Gold Standard:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(democrat_rouge_scores['rouge-1']),
                                                         np.mean(democrat_rouge_scores['rouge-2']),
                                                         np.mean(democrat_rouge_scores['rouge-l'])))
    print('############################')

    normal_rouge_scores = calculate_rouge_score(republican_gold_standards, normal_summaries)
    print('Normal Results against Republican Gold Standard:')
    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(normal_rouge_scores['rouge-1']),
                                                         np.mean(normal_rouge_scores['rouge-2']),
                                                         np.mean(normal_rouge_scores['rouge-l'])))
    print('############################')


def calculate_rouge_score(gold_standards, summaries):
    democrat_rouge_scores = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': []
    }
    for gold_standard in gold_standards:
        if len(gold_standard['content']) == 0:
            continue
        for summary in summaries:
            if summary['filename'] == gold_standard['filename']:
                rouge_scores = rouge.get_scores(summary['content'], gold_standard['content'])
                democrat_rouge_scores['rouge-1'].append(rouge_scores[0]['rouge-1']['f'])
                democrat_rouge_scores['rouge-2'].append(rouge_scores[0]['rouge-2']['f'])
                democrat_rouge_scores['rouge-l'].append(rouge_scores[0]['rouge-l']['f'])
    return democrat_rouge_scores


if __name__ == '__main__':
    main()
