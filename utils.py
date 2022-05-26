import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

def generate_full_shakespeare():
    headers = ["Player", "PlayerLine"]
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


def cleaning_with_space(text):
    word_buffer = ''
    delimiter = {'\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?'}
    filtered = []
    for i in text:
        if i in delimiter:
            if word_buffer != '':
                filtered.append(word_buffer)
            if i == "'":
                word_buffer = "'"
            else:
                filtered.append(i)
                word_buffer = ''
        else:
            word_buffer += i
    if word_buffer != '':
        filtered.append(word_buffer)
    return filtered


def cleaning_without_space(text):
    word_buffer = ''
    delimiter = {'\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?'}
    filtered = []
    for i in text:
        if i in delimiter:
            if word_buffer != '':
                filtered.append(word_buffer)
            if i == "'":
                word_buffer = "'"
            else:
                if i != ' ':
                    filtered.append(i)
                word_buffer = ''
        else:
            word_buffer += i
    if word_buffer != '':
        filtered.append(word_buffer)
    return filtered


def plot_metrics_correlation():
    headers = ["1-gram", "2-gram", "3-gram", "4-gram", "bleu3", "rouge-1-p", "rouge-1-r",
                                     "rouge-1-f1", "rouge-2-p", "rouge-2-r", "rouge-2-f1", "rouge-L-p", "rouge-L-r",
                                     "rouge-L-f1", "bert-p", "bert-r", "bert-f1", "bart", "zipf-coeff", "Human (0-3)"]
    df = pd.read_csv('eval_on_shakespeare_full - eval_on_shakespeare_full.csv', names=["out"] + headers)
    # df = pd.read_csv('eval_on_shakespeare_word_level_full - eval_on_shakespeare_word_level_full.csv', names=["out"] + headers)

    data = {}
    for header in headers:
        data[header] = [float(x) for x in df[header][1:]]

    print(data)
    df = pd.DataFrame(data, columns=headers)
    corrMatrix = df.corr(method='pearson')
    print(corrMatrix.shape)
    plt.figure(figsize=(9, 7))
    sn.heatmap(corrMatrix, annot=False)
    plt.title("metrics correlation")
    plt.show()

# plot_metrics_correlation()