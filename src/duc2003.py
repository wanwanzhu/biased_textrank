import re
from os import listdir

import nltk
import numpy as np
from biased_summarization import load_text_file, get_filenames_in_directory, get_sbert_embedding, biased_textrank, \
    select_top_k_texts_preserving_order
from rouge import Rouge

rouge = Rouge()


def read_duc_doc(path):
    text = load_text_file(path)
    text = text.split('<TEXT>')[1].split('</TEXT>')[0].strip().replace('<P>', '').replace('</P>', '')
    text = re.sub(r'``.*?\'\'', '.', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def read_task3_viewpoint(path):
    viewpoint = load_text_file(path)
    id = viewpoint.split('DOCSET="')[1].split('"')[0]
    viewpoint = viewpoint.split('\n\n')[1].split('</VIEWPOINT>')[0].strip()
    viewpoint = re.sub(r"\s+", ' ', viewpoint).strip()
    return id, viewpoint


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s and not s.isspace() and len(s) > 15]
    return sentences


def ls(path):
    return [filename for filename in listdir(path)]


def task3():
    clusters = load_task3_data()
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    for cluster in clusters:
        docs = cluster['docs']
        bias = cluster['viewpoint']
        references = cluster['references']

        doc_sentences = [sent for sent in get_sentences(doc) for doc in docs]
        doc_sentence_embeddings = get_sbert_embedding(doc_sentences)
        bias_embedding = get_sbert_embedding(bias)
        ranks = biased_textrank(doc_sentence_embeddings, bias_embedding, similarity_threshold=0.7)
        top_sentences = select_top_k_texts_preserving_order(doc_sentences, ranks, 10)
        summary = ''
        while len(nltk.word_tokenize(summary)) < 110:
            summary += top_sentences.pop(0) + ' '

        current_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        for reference in references:
            scores = rouge.get_scores(summary, reference)
            current_rouge_scores['rouge-1'].append(scores[0]['rouge-1']['f'])
            current_rouge_scores['rouge-2'].append(scores[0]['rouge-2']['f'])
            current_rouge_scores['rouge-l'].append(scores[0]['rouge-l']['f'])
        rouge_scores['rouge-1'].append(np.max(current_rouge_scores['rouge-1']))
        rouge_scores['rouge-2'].append(np.max(current_rouge_scores['rouge-2']))
        rouge_scores['rouge-l'].append(np.max(current_rouge_scores['rouge-l']))

    print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-l: {}'.format(np.mean(rouge_scores['rouge-1']),
                                                         np.mean(rouge_scores['rouge-2']),
                                                         np.mean(rouge_scores['rouge-l'])))


def load_task3_data():
    root_path = '../data/DUC03/task 3/'
    docs_path = root_path + 'docs/'
    viewpoints_path = root_path + 'viewpoints/'
    viewpoints_directories = ls(viewpoints_path)
    ground_truth_path = root_path + 'ground_truth/'
    ground_truth_filenames = get_filenames_in_directory(ground_truth_path)
    clusters = []
    for cluster_path in ls(docs_path):
        cluster_full_path = docs_path + cluster_path + '/'
        filepaths = get_filenames_in_directory(cluster_full_path)
        cluster_docs = []
        for filepath in filepaths:
            doc = read_duc_doc(cluster_full_path + filepath)
            cluster_docs.append(doc)

        viewpoint_path = [path for path in viewpoints_directories if path.startswith(cluster_path)][0]
        id, viewpoint = read_task3_viewpoint(viewpoints_path + viewpoint_path + '/vp')
        ground_truth_filename_prefix = id.upper()
        ground_truth_filepaths = [path for path in ground_truth_filenames if
                                  path.startswith(ground_truth_filename_prefix)]
        ground_truths = [load_text_file(ground_truth_path + _filepath) for _filepath in ground_truth_filepaths]

        clusters.append({
            'id': id,
            'docs': cluster_docs,
            'viewpoint': viewpoint,
            'references': ground_truths
        })

    return clusters


if __name__ == '__main__':
    task3()
