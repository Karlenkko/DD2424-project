import pandas as pd


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
