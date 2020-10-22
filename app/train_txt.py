import numpy as np
import pickle

from keras.models import Model
from keras.layers import Dense, GRU, Input, Masking
from keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt

import click


def make_dict(pkl_file):
    with open("utils/"+pkl_file, mode="rb") as chars:
        chars_list = pickle.load(chars)

    # インデックスと文字で辞書を作成
    char_indices = {}  # 文字がキーでインデックスが値
    for i, char in enumerate(chars_list):
        char_indices[char] = i
    indices_char = {}  # インデックスがキーで文字が値
    for i, char in enumerate(chars_list):
        indices_char[i] = char
    return chars_list, char_indices


def make_sentence(txt_file):
    with open("utils/"+txt_file, mode="r") as text:
        text = text.read()
    
    seperator = "。"
    sentence_list = text.split(seperator) 
    sentence_list.pop() 
    sentence_list = [x+seperator for x in sentence_list]

    max_sentence_length = 128
    sentence_list = [sentence for sentence in sentence_list 
                     if len(sentence) <= max_sentence_length]
    return sentence_list, max_sentence_length


def make_vector(chars_list, char_indices, sentence_list, max_sentence_length):
    n_char = len(chars_list)  # 文字の種類の数
    n_sample = len(sentence_list) - 1  # サンプル数

    x_sentences = []  # 入力の文章
    t_sentences = []  # 正解の文章
    for i in range(n_sample):
        x_sentences.append(sentence_list[i])
        t_sentences.append("\t" + sentence_list[i+1] + "\n")  # 正解は先頭にタブ、末尾に改行を加える
    max_length_x = max_sentence_length  # 入力文章の最大長さ
    max_length_t = max_sentence_length + 2  # 正解文章の最大長さ

    x_encoder = np.zeros((n_sample, max_length_x, n_char), dtype=np.bool)  # encoderへの入力
    x_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)  # decoderへの入力
    t_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)  # decoderの正解

    for i in range(n_sample):
        x_sentence = x_sentences[i]
        t_sentence = t_sentences[i]
        for j, char in enumerate(x_sentence):
            x_encoder[i, j, char_indices[char]] = 1  # encoderへの入力をone-hot表現で表す
        for j, char in enumerate(t_sentence):
            x_decoder[i, j, char_indices[char]] = 1  # decoderへの入力をone-hot表現で表す
            if j > 0:  # 正解は入力より1つ前の時刻のものにする
                t_decoder[i, j-1, char_indices[char]] = 1
            
    print(x_encoder.shape)
    return n_char, x_encoder, x_decoder, t_decoder


def train_txt(n_char, x_encoder, x_decoder, t_decoder):
    batch_size = 32
    epochs = 500
    n_mid = 256

    encoder_input = Input(shape=(None, n_char))
    encoder_mask = Masking(mask_value=0)
    encoder_masked = encoder_mask(encoder_input)
    encoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_state=True)
    encoder_output, encoder_state_h = encoder_lstm(encoder_masked)

    decoder_input = Input(shape=(None, n_char))
    decoder_mask = Masking(mask_value=0)
    decoder_masked = decoder_mask(decoder_input)
    decoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True)
    decoder_output, _ = decoder_lstm(decoder_masked, initial_state=encoder_state_h)
    decoder_dense = Dense(n_char, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    print(model.summary())

    early_stopping = EarlyStopping(monitor="val_loss", patience=30) 

    history = model.fit([x_encoder, x_decoder], t_decoder,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    encoder_model = Model(encoder_input, encoder_state_h)
    decoder_state_in_h = Input(shape=(n_mid,))
    decoder_state_in = [decoder_state_in_h]
    decoder_output, decoder_state_h = decoder_lstm(decoder_input,
                                                   initial_state=decoder_state_in_h)
    decoder_output = decoder_dense(decoder_output)
    decoder_model = Model([decoder_input] + decoder_state_in,
                          [decoder_output, decoder_state_h])
    # モデルの保存
    encoder_model.save('encoder_model.h5')
    decoder_model.save('decoder_model.h5')

    return loss, val_loss


def view_plot(loss, val_loss):
    plt.plot(np.arange(len(loss)), loss)
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.show()


@click.command()
@click.option('--pkl_file', '-p', default='prepare_text.pickle')
@click.option('--txt_file', '-t', default='prepare_text.txt')
def main(pkl_file, txt_file):
    chars_list, char_indices = make_dict(pkl_file)
    sentence_list, max_sentence_length = make_sentence(txt_file)
    n_char, x_encoder, x_decoder, t_decoder = make_vector(chars_list, char_indices, sentence_list, max_sentence_length)
    loss, val_loss = train_txt(n_char, x_encoder, x_decoder, t_decoder)
    view_plot(loss, val_loss)


if __name__ == "__main__":
    main()
