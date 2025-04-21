import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

FILLERS = ['uh', 'um', 'erm', 'hmm']

def count_fillers(text):
    return sum(text.lower().split().count(f) for f in FILLERS)

def extract_text_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    n_sentences = len(sentences)
    n_words = len(words)
    
    avg_sentence_len = n_words / n_sentences if n_sentences else 0
    pause_per_sentence = text.count("...") / n_sentences if n_sentences else 0
    
    return {
        "num_sentences": n_sentences,
        "num_words": n_words,
        "avg_sentence_len": avg_sentence_len,
        "filler_count": count_fillers(text),
        "pauses_per_sentence": pause_per_sentence,
    }
