import re
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import multiprocessing


def cleaning(text):
    word_buffer = ''
    delimiter = {'\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?'}
    filtered = ""
    for i in text:
        if i in delimiter:
            if word_buffer != '':
                filtered += " " + word_buffer
            if i == "'":
                word_buffer = "'"
            else:
                word_buffer = ''
        else:
            word_buffer += i
    return filtered

if __name__ == '__main__':
    headers = ["Player","PlayerLine"]
    df = pd.read_csv('Shakespeare_data.csv', names=headers)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    all = ""
    last_player = "????"
    for i in range(2, len(df)):
        if df["Player"][i] is None or df["Player"][i] == "" or df['PlayerLine'][i] is None or df['PlayerLine'][i] == "":
            continue

        if df["Player"][i] == last_player:
            all += df['PlayerLine'][i] + "\n"
        else:
            last_player = df["Player"][i]
            all += "\n" + last_player + ":\n"
            all += df['PlayerLine'][i] + "\n"

    with open("shakespeare_full.txt", "w") as text_file:
        text_file.write(all)
    exit(0)
    # print(cleaning("No more talking on't; let it be done: away, away!"))
    # brief_cleaning = (row.lower() for row in df['PlayerLine'])
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    txt = [row for row in df['PlayerLine']]

    # print(np.shape(txt))
    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    # # print(df_clean.shape)
    # sent = [row.split() for row in df_clean['clean']]
    # phrases = Phrases(sent, min_count=30, progress_per=10000)
    # bigram = Phraser(phrases)
    # sentences = bigram[sent]
    #
    # # word_freq = defaultdict(int)
    # # for sent in sentences:
    # #     for i in sent:
    # #         word_freq[i] += 1
    # # print(len(word_freq))
    # # print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])
    # cores = multiprocessing.cpu_count()
    # w2v_model = Word2Vec(min_count=1,
    #                      window=2,
    #                      sample=6e-5,
    #                      alpha=0.03,
    #                      min_alpha=0.0007,
    #                      negative=20,
    #                      workers=cores - 1)
    # t = time()
    #
    # w2v_model.build_vocab(sentences, progress_per=10000)
    #
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    # t = time()
    #
    # w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=1, report_delay=1)
    #
    # print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    #
    # w2v_model.wv.save("shakespeare_w2v.txt", pickle_protocol=0)