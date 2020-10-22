import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt


with open("./utils/sample.txt", mode="r", encoding="utf-8") as f:
    text_original = f.read()

text = re.sub("[　]", "", text_original)
print("文字数", len(text))

n_rnn = 8
batch_size = 128
epochs = 30
n_mid = 256

chars = sorted(list(set(text)))
print("文字数（重複無し）", len(chars))
char_indices = {}  # 文字がキーでインデックスが値
for i, char in enumerate(chars):
    char_indices[char] = i
indices_char = {}  # インデックスがキーで文字が値
for i, char in enumerate(chars):
    indices_char[i] = char
 
time_chars = []
next_chars = []
for i in range(0, len(text) - n_rnn):
    time_chars.append(text[i: i + n_rnn])
    next_chars.append(text[i + n_rnn])
 
x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
for i, t_cs in enumerate(time_chars):
    t[i, char_indices[next_chars[i]]] = 1
    for j, char in enumerate(t_cs):
        x[i, j, char_indices[char]] = 1
        
print("xの形状", x.shape)
print("tの形状", t.shape)


model = Sequential()
model.add(SimpleRNN(n_mid, input_shape=(n_rnn, len(chars))))
model.add(Dense(len(chars), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer="adam")
print(model.summary())

 
def on_epoch_end(epoch, logs):
    print("epoch number: ", epoch)

    beta = 5
    prev_text = text[0:n_rnn]
    created_text = prev_text
    
    print("シード: ", created_text)

    for i in range(100):
        # 入力をone-hot表現に
        x_pred = np.zeros((1, n_rnn, len(chars)))
        for j, char in enumerate(prev_text):
            x_pred[0, j, char_indices[char]] = 1
        
        # 予測を行い、次の文字を得る
        y = model.predict(x_pred)
        p_power = y[0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))        
        next_char = indices_char[next_index]

        created_text += next_char
        prev_text = prev_text[1:] + next_char

    print(created_text)
    print()

epock_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x, t,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[epock_end_callback])

loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()



# encoderのモデル
encoder_model = Model(encoder_input, encoder_state_h)

# decoderのモデル
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
