import os
import pickle


PICKLE_FILE = os.path.join('prepare_text.pickle')
TEXT_FILE = os.path.join('prepare_text.txt')


def make_dict():
    '''make dictionary using chars & keys'''
    with open(PICKLE_FILE, mode="rb") as chars:
        chars_list = pickle.load(chars)

    char_indices = {} 
    for i, char in enumerate(chars_list):
        char_indices[char] = i
    indices_char = {}
    for i, char in enumerate(chars_list):
        indices_char[i] = char
    return chars_list, char_indices


def make_sentence():
    '''make sentences by textfile'''
    with open(TEXT_FILE, mode="r") as text:
        text = text.read()
    
    seperator = "。"
    sentence_list = text.split(seperator) 
    sentence_list.pop() 
    sentence_list = [x+seperator for x in sentence_list]

    max_sentence_length = 128
    sentence_list = [sentence for sentence in sentence_list 
                     if len(sentence) <= max_sentence_length]
    return sentence_list, max_sentence_length


