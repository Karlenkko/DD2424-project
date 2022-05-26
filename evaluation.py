import os
import re

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *
from rouge import Rouge
import scipy.optimize as op
# os.environ['PATH'] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin" + os.pathsep + os.environ['PATH']
# print(os.environ['PATH'])
from jury import Jury


def get_references_tokenized(raw: str, start_token: str, length: int):
    occurrences = [m.start() for m in re.finditer(start_token, raw)]
    total_len = len(start_token) + length
    full = len(raw)
    ref = [cleaning_without_space(raw[i: min(full, i + total_len)]) for i in occurrences]
    return ref


def get_reference_strings(raw: str, start_token: str, length: int):
    occurrences = [m.start() for m in re.finditer(start_token, raw)]
    total_len = len(start_token) + length
    full = len(raw)
    ref = [raw[i: min(full, i + total_len)].lower() for i in occurrences]
    return ref


# raw is the full shakespeare text in string, to lower is enough
# out is the generated text, in string
# start is the start token used for generation
def calculate_bleu(raw, out, start, weights=(0.5, 0.5, 0, 0)):
    ref = get_references_tokenized(raw, start, len(out))
    score = nltk.bleu(references=ref, hypothesis=cleaning_without_space(out), weights=weights)
    return score


def test_bleu(raw):
    res = "romeo: not see tranio but him.\n\nmiranda:\nso frailty, thou he an action in rotten\nseize the wake humorous; and he ye iii.\nwhen stranger with such luke?\nignorant, stood, prevent would can these feast in\ngrow in  please?\nwould thy best."
    print(cleaning_without_space(res))
    ref = get_references_tokenized(raw, "romeo:", len(res))
    print(ref[0])
    score = nltk.bleu(references=ref, hypothesis=cleaning_without_space(res), weights=(0.5, 0.5, 0, 0))
    print(score)


def generate_ngrams(text: str, n=4):
    original = cleaning_with_space(text)  # 467259 tokens
    print(len(original))
    for i in range(1, n + 1):
        ngram = nltk.ngrams(original, i)
        freq = nltk.FreqDist(ngram)
        sorted_freq = []
        for item in freq.most_common():
            # print(item[0])
            sorted_freq += [["+".join([repr(str(token)) for token in item[0]]), int(item[1])]]
        pd.DataFrame(sorted_freq).to_csv(str(i) + "-gram_freq.csv", header=["token", "freq"], index=False)


def calculate_rouge(raw, out, start):
    ref = get_reference_strings(raw, start, len(out))
    rouge = Rouge()
    res = rouge.get_scores([out] * len(ref), ref, avg=True)
    return res
    # max_score = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
    #  'rouge-2': {'r': 0, 'p': 0, 'f': 0},
    #  'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    #
    # for item in res:
    #     max_score["rouge-1"]["r"] = max(max_score["rouge-1"]["r"], item["rouge-1"]["r"])
    #     max_score["rouge-1"]["p"] = max(max_score["rouge-1"]["p"], item["rouge-1"]["p"])
    #     max_score["rouge-1"]["f"] = max(max_score["rouge-1"]["f"], item["rouge-1"]["f"])


def calculate_zipf_coeff(out):
    tokenized = cleaning_without_space(out)
    ngram = nltk.ngrams(tokenized, 1)
    freq = nltk.FreqDist(ngram)
    tops = [count[1] for count in freq.most_common(10)]
    # tops = [26486, 12053, 5052, 3033, 2536, 2391, 1444, 1220, 1152, 1039]

    tops = np.array(tops) / np.sum(tops)

    def zipf_law(s):
        return -s * np.log(list(range(1, 11))) - np.log(np.sum(1 / np.power(np.array(list(range(1, 11))), s)))

    def cost(s: int): return np.sum(np.square(np.log(tops) - zipf_law(s)))

    res = op.minimize(cost, 1, method='BFGS')
    return res.x[0]


def test_rouge(raw):
    res = "romeo: not see tranio but him.\n\nmiranda:\nso frailty, thou he an action in rotten\nseize the wake humorous; and he ye iii.\nwhen stranger with such luke?\nignorant, stood, prevent would can these feast in\ngrow in  please?\nwould thy best."
    score = calculate_rouge(raw, res, "romeo:")
    print(score)


def calculate_bert_score(raw, out, start):
    ref = get_reference_strings(raw, start, len(out))

    score = scorer.evaluate(predictions=[out] * len(ref), references=ref)
    return score


def test_jury(raw):
    res = "romeo: not see tranio but him.\n\nmiranda:\nso frailty, thou he an action in rotten\nseize the wake humorous; and he ye iii.\nwhen stranger with such luke?\nignorant, stood, prevent would can these feast in\ngrow in  please?\nwould thy best."
    score = calculate_bert_score(raw, res, "romeo:")
    print(score)
    return score


def get_vocab(text: str):
    word_buffer = ''
    delimiter = {'\t', '\n', ' ', '!', '$', "'", '(', ')', ',', '-', '.', ':', '?', '[', ']'}
    vocab = set()
    # vocab = {word for word in delimiter}
    for i in text:
        if i in delimiter:
            if word_buffer != '' and word_buffer.lower() not in vocab:
                vocab.add(word_buffer.lower())
            if i == "'":
                word_buffer = "'"
            else:
                word_buffer = ''
        else:
            word_buffer += i
    if word_buffer != '' and word_buffer.lower() not in vocab:
        vocab.add(word_buffer.lower())
    for punkt in delimiter:
        if punkt in vocab:
            vocab.remove(punkt)
    return vocab


def calculate_spelling_accuracy(raw, out):
    original_vocab = get_vocab(raw)
    out_vocab = get_vocab(out)
    count = 0
    for token in out_vocab:
        if token in original_vocab:
            count += 1
    return count / len(out_vocab)

def gen_bart(raw, files):
    start = "romeo:"
    outputs = []
    out_dirs = []
    res = []
    for out_dir in files:
        out_dirs.append(out_dir)
        outputs.append(open(out_dir, mode='r').read())
    for out, dir in zip(outputs, out_dirs):
        print("evaluating " + dir)
        bert = calculate_bert_score(raw, out, start)
        bart = np.mean(bert['bartscore']["score"])
        scores = [dir, bart]
        res.append(scores)
    pd.DataFrame(res).to_csv("bart.csv",
                             header=["out", "bart"], index=False)

def gen_spelling_accuracy(raw, files):
    outputs = []
    out_dirs = []
    res = []
    for out_dir in files:
        out_dirs.append(out_dir)
        outputs.append(open(out_dir, mode='r').read())
    for out, dir in zip(outputs, out_dirs):
        scores = [dir, calculate_spelling_accuracy(raw, out)]
        res.append(scores)
    pd.DataFrame(res).to_csv("char_level_spelling_acc-2.csv",
                             header=["out", "acc"], index=False)


def gen_metrics(raw, files, title):
    start = "romeo:"
    outputs = []
    out_dirs = []
    res = []
    for out_dir in files:
        out_dirs.append(out_dir)
        outputs.append(open(out_dir, mode='r').read())

    for out, dir in zip(outputs, out_dirs):
        print("evaluating " + dir)
        unigram_score = calculate_bleu(raw, out, start, (1, 0, 0, 0))
        bigram_score = calculate_bleu(raw, out, start, (0, 1, 0, 0))
        trigram_score = calculate_bleu(raw, out, start, (0, 0, 1, 0))
        fourgram_score = calculate_bleu(raw, out, start, (0, 0, 0, 1))
        bleu_3 = calculate_bleu(raw, out, start, (0.4, 0.4, 0.2, 0))
        rouge = calculate_rouge(raw, out, start)
        rouge_1_f1 = rouge["rouge-1"]["f"]
        rouge_2_f1 = rouge["rouge-2"]["f"]
        rouge_l_f1 = rouge["rouge-l"]["f"]
        rouge_1_recall = rouge["rouge-1"]["r"]
        rouge_2_recall = rouge["rouge-2"]["r"]
        rouge_l_recall = rouge["rouge-l"]["r"]
        rouge_1_p = rouge["rouge-1"]["p"]
        rouge_2_p = rouge["rouge-2"]["p"]
        rouge_l_p = rouge["rouge-l"]["p"]
        bert = calculate_bert_score(raw, out, start)
        bert_precision = np.mean(bert['bertscore']["precision"])
        bert_recall = np.mean(bert['bertscore']["recall"])
        bert_f1 = np.mean(bert['bertscore']["f1"])
        bart = np.mean(bert['bartscore']["score"])
        zipf = calculate_zipf_coeff(out)
        scores = [dir, unigram_score, bigram_score, trigram_score, fourgram_score, bleu_3, rouge_1_p, rouge_1_recall,
                  rouge_1_f1, rouge_2_p, rouge_2_recall, rouge_2_f1, rouge_l_p, rouge_l_recall, rouge_l_f1,
                  bert_precision, bert_recall, bert_f1, bart, zipf]
        res.append(scores)
    pd.DataFrame(res).to_csv(title + ".csv",
                             header=["out", "1-gram", "2-gram", "3-gram", "4-gram", "bleu3", "rouge-1-p", "rouge-1-r",
                                     "rouge-1-f1", "rouge-2-p", "rouge-2-r", "rouge-2-f1", "rouge-L-p", "rouge-L-r",
                                     "rouge-L-f1", "bert-p", "bert-r", "bert-f1", "bart", "zipf-coeff"], index=False)


scorer = Jury(metrics=["bertscore", "bartscore"])
# scorer = Jury(metrics=["bartscore"])
if __name__ == '__main__':

    # outs = ["char_gru_100_full.txt", "char_gru_200_full.txt", "char_gru_300_full.txt", "char_gru_400_full.txt",
    #         "char_lstm_1_100_full.txt", "char_lstm_2_100_full.txt", "char_rnn_100_full.txt"]
    # outs += ["char_gru_500_full.txt", "char_gru_600_full.txt", "char_gru_800_full.txt", "char_gru_1000_full_60epochs.txt",
    #         "char_lstm_500_full_nu_085.txt", "char_lstm_500_full_nu_095.txt", "char_lstm_500_full_tem_08.txt", "char_lstm_500_full_tem_09.txt"]
    # outs = ["self_test.txt"]
    # outs = ["char_lstm_800_full_tem_09.txt", "char_lstm_1000_full_tem_09.txt", "char_lstm_2_500_full_tem_09.txt"]
    # outs = ["char_rnn_100_full.txt", "char_gru_100_full.txt", "char_gru_200_full.txt", "char_gru_300_full.txt", "char_gru_400_full.txt",
    #         "char_gru_500_full.txt", "char_gru_600_full.txt", "char_gru_800_full.txt", "char_gru_1000_full_60epochs.txt",
    #         "char_lstm_1_100_full.txt", "char_lstm_2_100_full.txt", "char_lstm_500_full_nu_085.txt", "char_lstm_500_full_nu_095.txt",
    #         "char_lstm_500_full_tem_08.txt", "char_lstm_500_full_tem_09.txt", "char_lstm_800_full_tem_09.txt", "char_lstm_1000_full_tem_09.txt",
    #         "char_lstm_2_500_full_tem_09.txt"]
    outs = ["char_lstm_200_full.txt", "char_lstm_400_full.txt", "char_lstm_500_full_nu_0999.txt", "char_lstm_500_full_nu_075.txt", "char_lstm_500_full_nu_065.txt", "char_lstm_500_full_tem_1.txt", "char_lstm_500_full_tem_07.txt"]
    # outs = ["word_lstm_500_glove_tem_09.txt", "word_lstm_500_ri_subset_norm_1_tem_09.txt", "word_lstm_500_ri_subset_norm_2_tem_09.txt", "word_lstm_500_w2v_tem_09.txt", "word_lstm_500_w2v_50_2_tem_09.txt"]
    for out_dir in outs:
        temp = open(out_dir, mode='r').read()

    # text = open('shakespeare.txt', mode='r').read()
    # test_rouge(text.lower())
    # gen_metrics(text.lower(), outs, "eval_on_shakespeare_subset")
    text = open('shakespeare_full.txt', mode='r').read()
    gen_spelling_accuracy(text.lower(), outs)
    # gen_metrics(text.lower(), outs, "eval_on_shakespeare_full_3")
    # gen_bart(text.lower(), outs)
    exit(0)
    # res = "romeo: not see tranio but him.\n\nmiranda:\nso frailty, thou he an action in rotten\nseize the wake humorous; and he ye iii.\nwhen stranger with such luke?\nignorant, stood, prevent would can these feast in\ngrow in  please?\nwould thy best."
    # print(cleaning_without_space(res))
    # ref = get_references_tokenized(text, "romeo:", len(res))
    # print(ref[0])
    # score = nltk.bleu(references=ref, hypothesis=cleaning_without_space(res), weights=(0.5, 0.5, 0, 0))
    # print(score)
