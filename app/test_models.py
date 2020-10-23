import os
import pickle
import numpy as np
from keras.models import load_model


ENCODER_MODEL = os.path.join('encoder_model.h5')
DECODER_MODEL = os.path.join('decoder_model.h5')
PICKLE_FILE = os.path.join('prepare_text.pickle')
TEXT_FILE = os.path.join('prepare_text.txt')


def respond(input_data, beta=5):
    encoder_model = load_model(ENCODER_MODEL)
    decoder_model = load_model(DECODER_MODEL)
    state_value = encoder_model.predict(input_data)
    y_decoder = np.zeros((1, 1, n_char))  # decoderの出力を格納する配列
    y_decoder[0][0][char_indices["\t"]] = 1  # decoderの最初の入力はタブ。one-hot表現にする。

    respond_sentence = ""  # 返答の文字列
    max_length_x = 140
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power)) 
        next_char = indices_char[next_index]  # 次の文字

        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了
            
        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))  # 次の時刻の入力
        y_decoder[0][0][next_index] = 1

        state_value = h  # 次の時刻の状態

    return respond_sentence


with open("utils/"+PICKLE_FILE, mode="rb") as chars:
    chars_list = pickle.load(chars)
    
n_char = len(chars_list) 
    # インデックスと文字で辞書を作成
char_indices = {}  # 文字がキーでインデックスが値
for i, char in enumerate(chars_list):
    char_indices[char] = i
indices_char = {}  # インデックスがキーで文字が値
for i, char in enumerate(chars_list):
    indices_char[i] = char


with open("utils/"+TEXT_FILE, mode="r") as text:
    text = text.read()
    
seperator = "。"
sentence_list = text.split(seperator) 
sentence_list.pop() 
sentence_list = [x+seperator for x in sentence_list]

max_sentence_length = 128
sentence_list = [sentence for sentence in sentence_list 
                 if len(sentence) <= max_sentence_length]

n_char = len(chars_list)
n_sample = len(sentence_list) - 1

x_sentences = []
t_sentences = []
for i in range(n_sample):
    x_sentences.append(sentence_list[i])
    t_sentences.append("\t" + sentence_list[i+1] + "\n")
max_length_x = max_sentence_length
max_length_t = max_sentence_length + 2

x_encoder = np.zeros((n_sample, max_length_x, n_char), dtype=np.bool)
x_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)
t_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)

for i in range(n_sample):
    x_sentence = x_sentences[i]
    t_sentence = t_sentences[i]
    for j, char in enumerate(x_sentence):
        x_encoder[i, j, char_indices[char]] = 1
    for j, char in enumerate(t_sentence):
        x_decoder[i, j, char_indices[char]] = 1
        if j > 0:
            t_decoder[i, j-1, char_indices[char]] = 1
            
print(x_encoder.shape)


for i in range(100):  
    x_in = x_encoder[i:i+1]  # 入力
    responce = respond(x_in)  # 返答
    print("Input:", x_sentences[i])
    print("Response:", responce)
    print()

