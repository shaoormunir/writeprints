def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import nltk
import spacy
import re
from sortedcontainers import SortedDict
import os
from keras.preprocessing import text

CHARACTER_FILTER = ' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"'
WORD_FILTER = ",.?!\"'`;:-()&$"

nlp = spacy.load('en_core_web_sm')
cur_dir_path = ""

def get_clean_text(text):
    # cleanText = text.text_to_word_sequence(text,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    clean_text = text.text_to_word_sequence(text, filters='', lower=False, split=" ")

    clean_text = ''.join(str(e) + " " for e in clean_text)
    return clean_text


def get_characters_count(text):
    '''
    Calculates character count including spaces
    '''
    text = text.lower()
    char_count = len(str(text))
    return char_count


def get_average_characters_per_word(text):
    '''
    Calculates average number of characters per word
    '''

    words = text.text_to_word_sequence(text, filters=CHARACTER_FILTER, lower=True, split=" ")
    text = text.lower().replace(" ", "")
    char_count = len(str(text))

    average_char_count = char_count / len(words)
    return average_char_count


def get_letters_frequency(text):
    '''
    Calculates the frequency of letters
    '''

    text = str(text).lower()  # because its case sensitive
    text = text.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    chars_frequency_dict = {}
    for c in range(0, len(characters)):
        char = characters[c]
        chars_frequency_dict[char] = 0
        for i in str(text):
            if char == i:
                chars_frequency_dict[char] = chars_frequency_dict[char] + 1

    # making a vector out of the frequencies
    frequency_vector = [0] * len(characters)
    total_count = sum(list(chars_frequency_dict.values()))
    for c in range(0, len(characters)):
        char = characters[c]
        frequency_vector[c] = chars_frequency_dict[char] / total_count

    return frequency_vector


def get_common_bigram_frequencies(text):
    # to do


    bigrams = ['th','he','in','er','an','re','nd','at','on','nt','ha','es','st' ,'en','ed','to','it','ou','ea','hi','is','or','ti','as','te','et' ,'ng','of','al','de','se','le','sa','si','ar','ve','ra','ld','ur']

    bigrams_dict = {}
    for t in bigrams:
        bigrams_dict[t] = True

    bigrams_counts_dict = {}
    for t in bigrams_dict:
        bigrams_counts_dict[t] = 0

    total_count = 0
    words = text.text_to_word_sequence(text, filters=WORD_FILTER, lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 1):
            bigram = str(word[i:i + 2]).lower()
            if bigram in bigrams:
                bigrams_counts_dict[bigram] = bigrams_counts_dict[bigram] + 1
                total_count = total_count + 1

    bigrams_frequencies = []
    for t in bigrams:
        bigrams_frequencies.append(float(bigrams_counts_dict[t] / total_count))

    return bigrams_frequencies


def get_common_trigram_frequencies(text):
    # to do
    trigrams = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for", "ent", "ion", "ter", "was", "you", "ith",
                "ver", "all", "wit", "thi", "tio"]
    trigrams_dict = {}
    for t in trigrams:
        trigrams_dict[t] = True

    trigrams_counts_dict = {}
    for t in trigrams_dict:
        trigrams_counts_dict[t] = 0

    total_count = 0
    words = text.text_to_word_sequence(text, filters=WORD_FILTER, lower=True, split=" ")
    for word in words:
        for i in range(0, len(word) - 2):
            trigram = str(word[i:i + 3]).lower()
            if trigram in trigrams:
                trigrams_counts_dict[trigram] = trigrams_counts_dict[trigram] + 1
                total_count = total_count + 1

    trigrams_frequencies = []
    for t in trigrams:
        trigrams_frequencies.append(float(trigrams_counts_dict[t] / total_count))

    return trigrams_frequencies


def get_digits_percentage(text):
    '''
    Calculates the percentage of digits out of total characters
    '''
    text = text.lower()
    chars_count = len(str(text))
    digits_count = list([1 for i in str(text) if i.isnumeric() == True]).count(1)
    return digits_count / chars_count


def get_characters_percentage(text):
    '''
    Calculates the percentage of characters out of total characters
    '''

    text = text.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    all_chars_count = len(str(text))
    chars_count = list([1 for i in str(text) if i in characters]).count(1)
    return chars_count / all_chars_count


def get_uppercase_characters_percentage(text):
    '''
    Calculates the percentage of uppercase characters out of total characters
    '''

    text = text.replace(" ", "")
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allchars_count = len(str(text))
    chars_count = list([1 for i in str(text) if i in characters]).count(1)
    return chars_count / allchars_count


def get_number_frequencies(text):
    '''
    Calculates the frequency of digits
    '''

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    digits_counts = {}
    for digit in digits:
        digits_counts[str(digit)] = 0

    alldigits = re.findall('\d', text)
    for digit in alldigits:
        digits_counts[digit] += 1

    digits_counts = SortedDict(digits_counts)
    digits_counts = np.array(digits_counts.values())
    return np.divide(digits_counts, get_characters_count(text))

def get_number_words_frequencies(text, digit_length):
    '''
    Calculates the frequency of digits
    '''

    text = str(text).lower()  # because its case sensitive
    words = text.text_to_word_sequence(text, filters=CHARACTER_FILTER, lower=True, split=" ")

    count = 0
    word_count = len(words)
    for w in words:
        if w.isnumeric() == True and len(w) == digit_length:
            count = count + 1

    return count / word_count


def get_word_length_frequencies(text):
    '''
    Calculate frequency of words of specific lengths upto 15
    '''
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    word_length_frequencies = {}
    for l in lengths:
        word_length_frequencies[l] = 0

    words = text.text_to_word_sequence(text, filters=CHARACTER_FILTER, lower=True, split=" ")
    for w in words:
        word_length = len(w)
        if word_length in word_length_frequencies:
            word_length_frequencies[word_length] = word_length_frequencies[word_length] + 1

    frequency_vector = [0] * (len(lengths))
    total_count = sum(list(word_length_frequencies.values()))
    for w in word_length_frequencies:
        frequency_vector[w - 1] = word_length_frequencies[w] / total_count

    return frequency_vector


def get_special_characters_frequencies(text):


    text = str(text).lower()  # because its case insensitive
    # text = text.lower().replace(" ", "")
    special_characters = open("resources/writeprints_special_chars.txt", "r").readlines()
    special_characters = [s.strip("\n") for s in special_characters]
    special_characters_dict = {}
    for c in range(0, len(special_characters)):
        special_character = special_characters[c]
        special_characters_dict[special_character] = 0
        for i in str(text):
            if special_character == i:
                special_characters_dict[special_character] = special_characters_dict[special_character] + 1


    # making a vector out of the frequencies
    frequency_vector = [0] * len(special_characters)
    total_count = sum(list(special_characters_dict.values())) + 1
    for c in range(0, len(special_characters)):
        special_character = special_characters[c]
        frequency_vector[c] = special_characters_dict[special_character] / total_count

    frequency_vector = np.array(frequency_vector)
    return frequency_vector


def get_function_words_percentage(text):
    function_words = open("resources/functionWord.txt", "r").readlines()
    function_words = [f.strip("\n") for f in function_words]
    # print((function_words))
    words = text.text_to_word_sequence(text, filters=CHARACTER_FILTER, lower=True, split=" ")
    function_words_frequencies = []
    for i in range(len(function_words)):
        functionWord = function_words[i]
        freq = 0
        for word in words:
            if word == functionWord:
                freq+=1
        function_words_frequencies.append(freq)
    # function_wordsIntersection = set(words).intersection(set(function_words))

    return function_words_frequencies


def get_punctuation_characters_frequencies(text):
    '''
    Calculates the frequency of special characters
    '''

    text = str(text).lower()  # because its case insensitive
    text = text.lower().replace(" ", "")
    special_characters = open("writeprintresources/writeprints_punctuation.txt", "r").readlines()
    special_characters = [s.strip("\n") for s in special_characters]
    special_characters_dict = {}
    for c in range(0, len(special_characters)):
        special_character = special_characters[c]
        special_characters_dict[special_character] = 0
        for i in str(text):
            if special_character == i:
                special_characters_dict[special_character] = special_characters_dict[special_character] + 1

    # making a vector out of the frequencies
    frequency_vector = [0] * len(special_characters)
    total_count = sum(list(special_characters_dict.values())) + 1
    for c in range(0, len(special_characters)):
        special_character = special_characters[c]
        frequency_vector[c] = special_characters_dict[special_character] / total_count

    return frequency_vector


def get_misspellings_percentage(text):
    misspelled_words = open("resources/writeprints_misspellings.txt", "r").readlines()
    misspelled_words = [f.strip("\n") for f in misspelled_words]
    words = text.text_to_word_sequence(text, filters=CHARACTER_FILTER, lower=True, split=" ")
    misspelled_words_intersection = set(words).intersection(set(misspelled_words))
    return len(misspelled_words_intersection) / len(list(words))


def legomena(text):
    freq = nltk.FreqDist(word for word in text.split())
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    try:
        return list((len(hapax) / len(text.split()),len(dis)/ len(text.split())))
    except:
        return [0,0]


def get_pos_tag_frequencies(text):
    pos_tags = []
    doc = nlp(str(text))
    for token in doc:
        pos_tags.append(str(token.pos_))

    # tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tagset = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
    tags = [tag for tag in pos_tags]
    return list(tuple(tags.count(tag) / len(tags) for tag in tagset))

def get_total_words(text):
    words = text.text_to_word_sequence(text, filters=WORD_FILTER, lower=True, split=" ")
    return len(words)

def get_average_word_length(text):
    words = text.text_to_word_sequence(text, filters=WORD_FILTER, lower=True, split=" ")
    lengths = []
    for word in words:
        lengths.append(len(word))
    return np.mean(lengths)

def get_short_words_count(text):
    words = text.text_to_word_sequence(text, filters=WORD_FILTER, lower=True, split=" ")
    short_words = []
    for word in words:
        if len(word) <= 3:
            short_words.append(word)
    return len(short_words)