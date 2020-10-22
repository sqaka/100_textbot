# -*- coding: utf-8 -*-

import pickle
import re

import click
from pykakasi import kakasi


def read_char(chars):
    '''read basic chars'''
    with open("txt_data/"+chars, mode="r", encoding="utf-8") as f:
        char_text = re.sub("[ 　\n]", "", f.read()) 
    return char_text


def read_file(tweets):
    '''read textfile from tweets'''
    tweet_text = ""
    for tweet in tweets:
        with open("txt_data/"+tweets, mode="r", encoding="utf-8") as f:
            tweet = re.sub("[ 　\n「」『』（）｜※＊…]", "", f.read())
            tweet_text += tweet
        return tweet_text


def change_char(tweet_text, kakasi):
    '''change tweet_text, Kanji -> Hiragana'''
    seperator = "。"
    sentence_list = tweet_text.split(seperator)
    sentence_list.pop()
    sentence_list = [x+seperator for x in sentence_list]
    
    kakasi = kakasi()
    kakasi.setMode("J", "H")  # J(漢字) からH(ひらがな)へ
    conv = kakasi.getConverter()
    
    for sentence in sentence_list:
        print(sentence)
        print(conv.do(sentence))
        print()
        
    kana_text = conv.do(tweet_text)
    with open("prepare_text.txt", mode="w", encoding="utf-8") as f:
        f.write(kana_text)
    return kana_text


def make_corpus(kana_text, char_text):
    '''add any other char used in kana_text to char_text'''
    for char in kana_text:
        if char not in char_text:
            char_text += char
        
    char_text += "\t\n"     
    chars_list = sorted(list(char_text))
    print(chars_list)
    return chars_list


def save_pickle(chars_list):
    '''save prepare_text using pickle'''
    with open("prepare_text.pickle", mode="wb") as f:  # pickleで保存
        pickle.dump(chars_list, f)
        print("\ncomplete!!\n")


@click.command()
@click.option('--tweets', '-t', default='100.txt')
@click.option('--chars', '-c', default='chars.txt')
def main(tweets, chars):
    char_text = read_char(chars)
    tweet_text = read_file(tweets)
    kana_text = change_char(tweet_text, kakasi)
    chars_list = make_corpus(kana_text, char_text)
    save_pickle(chars_list)


if __name__ == "__main__":
    main()
