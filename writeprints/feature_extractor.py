import numpy as np
import nltk
import spacy
import re
from sortedcontainers import SortedDict
import os
from keras.preprocessing import text as tx
from pkg_resources import resource_filename


class FeatureExtractor(object):
    def __init__(self, flatten):
        self.CHARACTER_FILTER = ' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"'
        self.WORD_FILTER = ",.?!\"'`;:-()&$"
        self.nlp = spacy.load('en_core_web_sm')
        self.flatten = flatten

    def get_clean_text(self, text):
        clean_text = tx.text_to_word_sequence(
            text, filters='', lower=False, split=" ")

        clean_text = ''.join(str(e) + " " for e in clean_text)
        return clean_text

    def get_characters_count(self, text):
        '''
        Calculates character count including spaces
        '''
        feature_labels = []
        feature_labels.append('characters_count')
        text = text.lower()
        char_count = len(str(text))
        return char_count, feature_labels

    def get_average_characters_per_word(self, text):
        '''
        Calculates average number of characters per word
        '''
        feature_labels = []
        feature_labels.append('average_characters_per_word')

        words = tx.text_to_word_sequence(
            text, filters=self.CHARACTER_FILTER, lower=True, split=" ")
        text = text.lower().replace(" ", "")
        char_count = len(str(text))
        try:
            average_char_count = char_count / len(words)
        except ZeroDivisionError:
            average_char_count = float('NaN')
        return average_char_count, feature_labels

    def get_letters_frequency(self, text):
        '''
        Calculates the frequency of letters
        '''
        feature_labels = []

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
            feature_labels.append(f"letters_frequency:{char}")
            try:
                frequency_vector[c] = chars_frequency_dict[char] / total_count
            except ZeroDivisionError:
                frequency_vector[c] = float('NaN')

        return frequency_vector, feature_labels

    def get_common_bigram_frequencies(self, text):
        feature_labels = []

        bigrams = ['th', 'he', 'in', 'er', 'an', 're', 'nd', 'at', 'on', 'nt', 'ha', 'es', 'st', 'en', 'ed', 'to', 'it', 'ou', 'ea',
                   'hi', 'is', 'or', 'ti', 'as', 'te', 'et', 'ng', 'of', 'al', 'de', 'se', 'le', 'sa', 'si', 'ar', 've', 'ra', 'ld', 'ur']

        bigrams_dict = {}
        for t in bigrams:
            bigrams_dict[t] = True

        bigrams_counts_dict = {}
        for t in bigrams_dict:
            bigrams_counts_dict[t] = 0

        total_count = 0
        words = tx.text_to_word_sequence(
            text, filters=self.WORD_FILTER, lower=True, split=" ")
        for word in words:
            for i in range(0, len(word) - 1):
                bigram = str(word[i:i + 2]).lower()
                if bigram in bigrams:
                    bigrams_counts_dict[bigram] = bigrams_counts_dict[bigram] + 1
                    total_count = total_count + 1

        bigrams_frequencies = []
        for t in bigrams:
            feature_labels.append(f'common_bigram_frequencies:{t}')
            try:
                bigrams_frequencies.append(
                    float(bigrams_counts_dict[t] / total_count))
            except ZeroDivisionError:
                bigrams_frequencies.append(
                    float('NaN'))

        return bigrams_frequencies, feature_labels

    def get_common_trigram_frequencies(self, text):
        # to do

        feature_labels = []
        trigrams = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for", "ent", "ion", "ter", "was", "you", "ith",
                    "ver", "all", "wit", "thi", "tio"]
        trigrams_dict = {}
        for t in trigrams:
            trigrams_dict[t] = True

        trigrams_counts_dict = {}
        for t in trigrams_dict:
            trigrams_counts_dict[t] = 0

        total_count = 0
        words = tx.text_to_word_sequence(
            text, filters=self.WORD_FILTER, lower=True, split=" ")
        for word in words:
            for i in range(0, len(word) - 2):
                trigram = str(word[i:i + 3]).lower()
                if trigram in trigrams:
                    trigrams_counts_dict[trigram] = trigrams_counts_dict[trigram] + 1
                    total_count = total_count + 1

        trigrams_frequencies = []
        for t in trigrams:
            feature_labels.append(f'common_trigram_frequencies:{t}')
            try:
                trigrams_frequencies.append(
                    float(trigrams_counts_dict[t] / total_count))
            except ZeroDivisionError:
                trigrams_frequencies.append(
                    float('NaN'))

        return trigrams_frequencies, feature_labels

    def get_digits_percentage(self, text):
        '''
        Calculates the percentage of digits out of total characters
        '''
        feature_labels = []
        feature_labels.append('digits_percentage')
        text = text.lower()
        chars_count = len(str(text))
        digits_count = list(
            [1 for i in str(text) if i.isnumeric() == True]).count(1)
        try:
            return digits_count / chars_count, feature_labels
        except ZeroDivisionError:
            return float('NaN'), feature_labels

    def get_characters_percentage(self, text):
        '''
        Calculates the percentage of characters out of total characters
        '''
        feature_labels = []
        feature_labels.append('characters_percentage')

        text = text.lower().replace(" ", "")
        characters = "abcdefghijklmnopqrstuvwxyz"
        all_chars_count = len(str(text))
        chars_count = list([1 for i in str(text) if i in characters]).count(1)
        try:
            return chars_count / all_chars_count, feature_labels
        except ZeroDivisionError:
            return float('NaN'), feature_labels

    def get_uppercase_characters_percentage(self, text):
        '''
        Calculates the percentage of uppercase characters out of total characters
        '''
        feature_labels = []
        feature_labels.append('uppercase_characters_percentage')
        text = text.replace(" ", "")
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        allchars_count = len(str(text))
        chars_count = list([1 for i in str(text) if i in characters]).count(1)
        try:
            return chars_count / allchars_count, feature_labels
        except ZeroDivisionError:
            return float('NaN'), feature_labels

    def get_number_frequencies(self, text):
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
        feature_labels = [
            f'number_frequencies:{x}' for x in digits_counts.keys()]
        digits_counts = np.array(digits_counts.values())
        character_counts, _ = self.get_characters_count(text)
        digits_counts = np.divide(digits_counts, character_counts)

        return digits_counts.tolist(), feature_labels

    def get_number_words_frequencies(self, text, digit_length):
        '''
        Calculates the frequency of digits
        '''
        feature_labels = []
        feature_labels.append("number_words_frequencies")

        text = str(text).lower()  # because its case sensitive
        words = tx.text_to_word_sequence(
            text, filters=self.CHARACTER_FILTER, lower=True, split=" ")

        count = 0
        word_count = len(words)
        for w in words:
            if w.isnumeric() == True and len(w) == digit_length:
                count = count + 1

        try:
            return count / word_count, feature_labels
        except ZeroDivisionError:
            return float('NaN'), feature_labels

    def get_word_length_frequencies(self, text):
        '''
        Calculate frequency of words of specific lengths upto 15
        '''

        lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        feature_labels = []
        word_length_frequencies = {}
        for l in lengths:
            word_length_frequencies[l] = 0

        words = tx.text_to_word_sequence(
            text, filters=self.CHARACTER_FILTER, lower=True, split=" ")
        for w in words:
            word_length = len(w)
            if word_length in word_length_frequencies:
                word_length_frequencies[word_length] = word_length_frequencies[word_length] + 1

        frequency_vector = [0] * (len(lengths))
        total_count = sum(list(word_length_frequencies.values()))
        for w in word_length_frequencies:
            feature_labels.append(f"word_length_frequencies:{w}")
            try:
                frequency_vector[w -
                                 1] = word_length_frequencies[w] / total_count
            except ZeroDivisionError:
                frequency_vector[w - 1] = float('NaN')

        return frequency_vector, feature_labels

    def get_special_characters_frequencies(self, text):

        feature_labels = []

        text = str(text).lower()
        special_characters = open(resource_filename(
            'writeprints', 'writeprintresources/writeprints_special_chars.txt')).readlines()
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
            feature_labels.append(
                f"special_characters_frequencies:{special_character}")
            try:
                frequency_vector[c] = special_characters_dict[special_character] / total_count
            except ZeroDivisionError:
                frequency_vector[c] = float('NaN')

        frequency_vector = np.array(frequency_vector)
        return frequency_vector.tolist(), feature_labels

    def get_function_words_frequencies(self, text):
        feature_labels = []
        function_words = open(resource_filename(
            'writeprints', 'writeprintresources/functionWord.txt')).readlines()
        function_words = [f.strip("\n") for f in function_words]
        # print((function_words))
        words = tx.text_to_word_sequence(
            text, filters=self.CHARACTER_FILTER, lower=True, split=" ")
        function_words_frequencies = []
        for i in range(len(function_words)):
            function_word = function_words[i]
            feature_labels.append(
                f"function_words_frequencies:{function_word}")
            freq = 0
            for word in words:
                if word == function_word:
                    freq += 1
            function_words_frequencies.append(freq)
        function_words_frequencies, feature_labels = zip(
            *sorted(zip(function_words_frequencies, feature_labels), reverse=True))
        function_words_frequencies = function_words_frequencies[:50]
        feature_labels = feature_labels[:50]

        return list(function_words_frequencies), list(feature_labels)

    def get_punctuation_characters_frequencies(self, text):
        '''
        Calculates the frequency of special characters
        '''
        feature_labels = []

        text = str(text).lower()  # because its case insensitive
        text = text.lower().replace(" ", "")
        special_characters = open(resource_filename(
            'writeprints', 'writeprintresources/writeprints_punctuation.txt')).readlines()
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
            feature_labels.append(
                f"punctuation_characters_frequencies:{special_character}")
            try:
                frequency_vector[c] = special_characters_dict[special_character] / total_count
            except ZeroDivisionError:
                frequency_vector[c] = float('NaN')

        return frequency_vector, feature_labels

    def get_misspellings_percentage(self, text):
        feature_labels = []
        feature_labels.append("misspellings_percentage")
        misspelled_words = open(resource_filename(
            'writeprints', 'writeprintresources/writeprints_misspellings.txt')).readlines()
        misspelled_words = [f.strip("\n") for f in misspelled_words]
        words = tx.text_to_word_sequence(
            text, filters=self.CHARACTER_FILTER, lower=True, split=" ")
        misspelled_words_intersection = set(
            words).intersection(set(misspelled_words))
        try:
            return len(misspelled_words_intersection) / len(list(words)), feature_labels
        except ZeroDivisionError:
            return float('NaN'), feature_labels

    def legomena(self, text):
        feature_labels = []
        feature_labels.append("leogomena:1")
        feature_labels.append("leogomena:2")
        freq = nltk.FreqDist(word for word in text.split())
        hapax = [key for key, val in freq.items() if val == 1]
        dis = [key for key, val in freq.items() if val == 2]
        try:
            return list((len(hapax) / len(text.split()), len(dis) / len(text.split()))), feature_labels
        except ZeroDivisionError:
            return [0, 0], feature_labels

    def get_pos_tag_frequencies(self, text):
        pos_tags = []
        doc = self.nlp(str(text))
        for token in doc:
            pos_tags.append(str(token.pos_))

        tagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                  'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
        tags = [tag for tag in pos_tags]
        feature_labels = [f'pos_tag_frequencies:{x}' for x in tagset]
        try:
            return [tags.count(tag) / len(tags) for tag in tagset], feature_labels
        except ZeroDivisionError:
            return [float('NaN') for tag in tagset], feature_labels

    def get_total_words(self, text):
        feature_labels = []
        feature_labels.append('total_words')
        words = tx.text_to_word_sequence(
            text, filters=self.WORD_FILTER, lower=True, split=" ")
        return len(words), feature_labels

    def get_average_word_length(self, text):
        feature_labels = []
        feature_labels.append('average_word_length')
        words = tx.text_to_word_sequence(
            text, filters=self.WORD_FILTER, lower=True, split=" ")
        lengths = []
        for word in words:
            lengths.append(len(word))
        return np.mean(lengths), feature_labels

    def get_short_words_count(self, text):
        feature_labels = []
        feature_labels.append('short_word_count')
        words = tx.text_to_word_sequence(
            text, filters=self.WORD_FILTER, lower=True, split=" ")
        short_words = []
        for word in words:
            if len(word) <= 3:
                short_words.append(word)
        return len(short_words), feature_labels

    def process(self, text):

        features_dict = {"characters_count": self.get_characters_count, "average_characters_per_word": self.get_average_characters_per_word, "letters_frequency": self.get_letters_frequency, "common_bigram_frequencies": self.get_common_bigram_frequencies, "common_trigram_frequencies": self.get_common_trigram_frequencies, "digits_percentage": self.get_digits_percentage, "characters_percentage": self.get_characters_percentage, "uppercase_characters_percentage": self.get_uppercase_characters_percentage, "number_frequencies": self.get_number_frequencies,
                         "word_length_frequencies": self.get_word_length_frequencies, "special_characters_frequencies": self.get_special_characters_frequencies, "function_words_frequencies": self.get_function_words_frequencies, "punctuation_characters_frequencies": self.get_punctuation_characters_frequencies, "misspellings_percentage": self.get_misspellings_percentage, "leogomena": self.legomena, "pos_tag_frequencies": self.get_pos_tag_frequencies, "total_words": self.get_total_words, "average_word_length": self.get_average_word_length, "short_word_count": self.get_short_words_count}

        output_dict = {}

        for feature in features_dict:
            features, feature_labels = features_dict[feature](text)
            if self.flatten and not(isinstance(features, int) or isinstance(features, float)):
                for f, l in zip(features, feature_labels):
                    output_dict[l] = f
            else:
                output_dict[feature] = features

        return output_dict
