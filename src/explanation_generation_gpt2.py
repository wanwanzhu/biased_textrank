import json
import random
import sys

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import gpt_2_simple as gpt2
import tensorflow as tf

from biased_summarization import select_top_k_texts_preserving_order, biased_textrank, get_sbert_embedding
from explanation_generation import get_sentences


def almost_the_same(a, b):
    len_ratio = len(a) / len(b) if len(a) < len(b) else len(b) / len(a)
    similarity = fuzz.partial_ratio(a, b)
    return len_ratio >= 0.9 and similarity > 90


def generated_text_is_meaningful(text, generation_prefix):
    return text != '' and not text.isspace() and not almost_the_same(text, generation_prefix)


def get_generation_prefix(article, claim_statement):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article + '\n'
    generation_prefix += '<|CLAIM|>: ' + claim_statement + '\n'
    generation_prefix += '<|EXPLANATION|>: '
    return generation_prefix


def generate_explanation(article, question, session):
    generation_prefix = get_generation_prefix(article, question)
    temperature = 0.7
    while True:
        generated_explanations = gpt2.generate(session, prefix=generation_prefix, truncate='<|endoftext|>', length=80,
                                               include_prefix=False, temperature=temperature, return_as_list=True, batch_size=2,
                                               nsamples=2, run_name='simple')
        for generated_explanation in generated_explanations:
            if generated_text_is_meaningful(generated_explanation, generation_prefix) or temperature >= 0.8:
                print(generated_explanation)
                return generated_explanation

        temperature += 0.1


def fine_tune_gpt2():
    MODEL_NAME = '355M'
    TRAINING_DATA_PATH = '../data/liar/gpt2_training_data.txt'
    session = gpt2.start_tf_sess()
    gpt2.finetune(session, TRAINING_DATA_PATH, model_name=MODEL_NAME, steps=400, run_name='simple')


def generate_explanations_using_gpt2(split):
    data_points_summarized = 0
    session = gpt2.start_tf_sess()
    gpt2.load_gpt2(session, run_name='simple', multi_gpu=True)
    with open('../data/liar/clean_{}.json'.format(split)) as input_file:
        dataset = json.load(input_file)
    for claim_id, claim in enumerate(dataset):
        statements = claim['statements']

        if 'generated_justification_gpt2' in claim and generated_text_is_meaningful(claim['generated_justification_gpt2'], get_generation_prefix(statements, claim['claim'])):
            print('Skipping item #{} because it already has a meaningful generated explanation.'.format(claim_id))
            continue

        summary_size = min(20, len(get_sentences(claim['statements'])) - 2)
        summary_doesnt_fit = True
        while summary_doesnt_fit:
            try:
                print('Generating explanation for article #{} ...'.format(claim_id))
                claim['generated_justification_gpt2'] = generate_explanation(statements, claim['claim'], session)
                if generated_text_is_meaningful(claim['generated_justification_gpt2'], get_generation_prefix(statements, claim['claim'])):
                    summary_doesnt_fit = False
                elif summary_size <= 10:
                    claim['generated_justification_gpt2'] = ''
                else:
                    print('Generated explanation for item #{} was not meaningful.'.format(claim_id))
                    raise ValueError(
                        'Generated explanation was gibberish (whitespace or repeating precondition text)')
            except Exception as e:
                print(e)
                if summary_size == 20:  # gotta make sure we only increment this once per article at most
                    data_points_summarized += 1

                print('Running biased textrank for item #{} ...'.format(claim_id))
                statements_sentences = get_sentences(claim['statements'])
                statements_embeddings = get_sbert_embedding(statements_sentences)
                bias = claim['claim']
                bias_embedding = get_sbert_embedding(bias)
                ranking = biased_textrank(statements_embeddings, bias_embedding)
                print('Biased textrank completed.')
                top_sentences = select_top_k_texts_preserving_order(statements, ranking, summary_size)
                statements_summary = ' '.join(top_sentences)
                statements = statements_summary
                summary_size -= 2

        with open('../data/liar/_{}.json'.format(split), 'w') as f:
            f.write(json.dumps(dataset))
        print('results for {} set saved. Data points summarized so far: {}'.format(split, data_points_summarized))

        # K.clear_session()
        if claim_id % 20 == 0:  # bug fix for slow down in generation
            tf.reset_default_graph()
            session = gpt2.start_tf_sess()
            gpt2.load_gpt2(session, run_name='simple', multi_gpu=True)

    tf.reset_default_graph()

    with open('../data/liar/_{}.json'.format(split), 'w') as f:
        f.write(json.dumps(dataset))

    print('all explanations generated, total summarized are: {}.'.format(data_points_summarized))


def generate_training_string(claims):
    train_str = ''
    for claim in claims:
        if claim['new_justification'] == '' or claim['new_justification'].isspace() or len(claim['new_justification']) < 10:
            continue
        train_str += '<|startoftext|>' + '\n'
        train_str += claim['statements'] + '\n'
        train_str += '<|CLAIM|>>: ' + claim['claim'] + '\n'
        train_str += '<|EXPLANATION|>: ' + claim['new_justification'] + '\n'
        train_str += '<|endoftext|>' + '\n'
    return train_str


def prepare_training_data_for_gpt2():
    with open('../data/liar/clean_train.json') as input_file:
        dataset = json.load(input_file)

    random.Random(2017).shuffle(dataset)

    train_str = generate_training_string(dataset)

    with open('../data/liar/gpt2_training_data.txt', 'w') as f:
        f.write(train_str)


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else ''
    # prepare_training_data_for_gpt2()
    # fine_tune_gpt2()
    generate_explanations_using_gpt2(split)
