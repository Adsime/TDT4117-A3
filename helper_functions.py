import string
from nltk.stem.porter import PorterStemmer
import codecs

stemmer = PorterStemmer()


def get_paragraphs(file):
    paragraph = ""
    paragraphs = []
    for line in file.readlines():
        if line.isspace():
            if paragraph != "":
                paragraphs.append(paragraph)
            paragraph = ""
            continue
        paragraph += line
    return paragraphs


def remove(word, paragraphs):
    for p in paragraphs:
        if p.__contains__(word):
            paragraphs.remove(p)
    return paragraphs


def tokenize(paragraphs):
    for i, p in enumerate(paragraphs):
        paragraphs[i] = tokenize_string(p)
    return paragraphs


def tokenize_string(words):
    return words.split(" ")


def remove_punctuations(paragraphs):
    for i, word_list in enumerate(paragraphs):
        words = remove_punctuations_in_list(word_list)
        paragraphs[i] = words
    return paragraphs


def remove_punctuations_in_list(list):
    words = []
    for word in list:
        w = ""
        for letter in word:
            if (string.punctuation + "\n\r\t").__contains__(letter):
                if w != "":
                    words.append(w.lower())
                    w = ""
                continue
            w += letter
        if w != "":
            words.append(w.lower())
    return words


def stem(paragraphs):
    for i, p in enumerate(paragraphs):
            paragraphs[i] = stem_list(p)
    return paragraphs


def stem_list(list):
    for i, word in enumerate(list):
        list[i] = stemmer.stem(word.lower())
    return list


def get_stop_words(file):
    stopwords = []
    for line in file.readlines():
        for word in line.split(","):
            stopwords.append(word)
    return stopwords


def get_stop_wordids(stopwords, dictionary):
    stopIds = []
    for word in stopwords:
        try:
            stopIds.append(dictionary.token2id[word])
        except:
            pass
    return stopIds

def process(query):
    query = tokenize_string(query)
    query = remove_punctuations_in_list(query)
    query = stem_list(query)
    return query
