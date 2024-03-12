from glob import glob
from vectorize import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

EQUALIZED = True
NEGATIVE_PREFIX = "./data/negatives/examples/"
POSITIVE_PREFIX = "./data/positives/{}/".format("equalized_length" if EQUALIZED else "examples")
NEGATIVE_FORM = NEGATIVE_PREFIX + "{}.txt"
POSITIVE_FORM = POSITIVE_PREFIX + "{}.txt"
EXCLUDE_INCOMPLETE = False
DO_MASK = True
MASK_PHRASES = ["In conclusion", "Overall"]

negatives = glob(NEGATIVE_PREFIX + "*.txt")
positives = glob(POSITIVE_PREFIX + "*.txt")
negative_names = [i.split("/")[-1].split(".")[0] for i in negatives]
if EXCLUDE_INCOMPLETE:
    _n = []
    for i in negative_names:
        with open(i, "r") as doc:
            if "."  in doc.read()[-5:]:
                _n.append(i)
    negative_names = _n
positive_names = [i.split("/")[-1].split(".")[0] for i in positives]

matches = list(filter(lambda e: e in negative_names, positive_names))
matched_negatives = [NEGATIVE_FORM.format(i) for i in matches]
matched_positives = [POSITIVE_FORM.format(i) for i in matches]

class Dataset(object):
    def __init__(self, negatives, positives, vectorizer, train_size=.8, split=True, paragraph=False, mask=DO_MASK, mask_phrases=MASK_PHRASES, no_vec=False, sentence=False):
        self.vectorizer = vectorizer()
        self.train_size = train_size
        self.negative_files = negatives
        self.positive_files = positives
        self.paragraph = paragraph
        self.mask = mask
        self.mask_phrases = mask_phrases
        self.no_vec = no_vec
        self.sentence = sentence

        self._isolate_names()
        self._compile_texts()
        self._label_data()
        
        if not no_vec:
            self._vectorize_data()
        self._isolate_xy()
        
        if split:
            self._train_test()
        else:
            if not no_vec:
                self.x_train = self.x
                self.y_train = self.y
                self.x_test = self.x
                self.y_test = self.y
            self.x_train_text = self.x_text
            self.y_train_text = self.y_text
            self.x_test_text = self.x_text
            self.y_test_text = self.y_text

    def _isolate_names(self):
        self.negative_names = [i.split("/")[-1].split(".")[0] for i in self.negative_files]
        self.positive_names = [i.split("/")[-1].split(".")[0] for i in self.positive_files]
    
    def _compile_texts(self):
        self.negative_texts = []
        for i in self.negative_files:
            with open(i, "r") as doc:
                self.negative_texts.append(doc.read())
        
        self.positive_texts = []
        for i in self.positive_files:
            with open(i, "r") as doc:
                self.positive_texts.append(doc.read())
        
        if self.mask:
            _n = []
            for i in self.negative_texts:
                for n in self.mask_phrases:
                    i = i.replace(n, "")
                _n.append(i)
            self.negative_texts = _n
            _n = []
            for i in self.positive_texts:
                for n in self.mask_phrases:
                    i = i.replace(n, "")
                _n.append(i)
            self.positive_texts = _n

        if self.paragraph:
            pt = []
            nt = []
            for i in self.positive_texts:
                pt += i.split("\n")
            for i in self.negative_texts:
                nt += i.split("\n")
            self.positive_texts = pt
            self.negative_texts = nt
        
        if self.sentence:
            pt = []
            nt = []
            for i in self.positive_texts:
                pt += i.replace(". ", ".").split(".")
            for i in self.negative_texts:
                nt += i.replace(". ", ".").split(".")
            self.positive_texts = pt
            self.negative_texts = nt
        self.positive_texts = list(filter(lambda e: len(e.replace("\n", "").replace(" ", "").replace(".", "").replace(",", "")) > 0, self.positive_texts))
        self.negative_texts = list(filter(lambda e: len(e.replace("\n", "").replace(" ", "").replace(".", "").replace(",", "")) > 0, self.negative_texts))
    
    def _label_data(self):
        self.file_lookup = {**{n: 0 for n in self.negative_files}, **{p: 1 for p in self.positive_files}}
        self.text_data = {**{n: 0 for n in self.negative_texts}, **{p: 1 for p in self.positive_texts}}

    def _vectorize_data(self):
        self.text_to_vector = {text: vec for (vec, text) in zip(self.vectorizer.fit_transform(self.text_data).toarray().tolist(), self.text_data)}
        self.vectorized_data = [(self.text_to_vector[i], self.text_data[i]) for i in self.text_data]
        self.vectorized_negatives = [vec for (vec, val) in self.vectorized_data if val == 0]
        self.vectorized_positives = [vec for (vec, val) in self.vectorized_data if val == 1]
        self.vectors_by_val = {0: self.vectorized_negatives, 1: self.vectorized_positives}
    
    def _isolate_xy(self):
        if not self.no_vec:
            (self.x, self.y) = ([i[0] for i in self.vectorized_data], [i[1] for i in self.vectorized_data])
        (self.x_text, self.y_text) = ([text for (text, _) in self.text_data.items()], [val for (_, val) in self.text_data.items()])
    
    def _train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=self.train_size)
        self.x_train_text, self.x_test_text, self.y_train_text, self.y_test_text = train_test_split(self.x_text, self.y_text, train_size=self.train_size)
    
    def vectorized_fit(self):
        return (self.x_train, self.y_train)
    
    def text_fit(self):
        return (self.x_train_text, self.y_train_text)
    
    def output(self, path, text=True):
        if text:
            pd.DataFrame([(n, i) for (i, n) in self.text_data.items()], columns=["label", "text"]).to_csv(path)
        else:
            print("No Text Not Implemented")

def TFIDF_matched_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, TfidfVectorizer, split=split, no_vec=no_vec)
def BOW_matched_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, CountVectorizer, split=split, no_vec=no_vec)

def TFIDF_full_dataset(split=True, no_vec=False):
    return Dataset(negatives, positives, TfidfVectorizer, split=split, no_vec=no_vec)
def BOW_full_dataset(split=True, no_vec=False):
    return Dataset(negatives, positives, CountVectorizer, split=split, no_vec=no_vec)

def TFIDF_matched_paragraph_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, TfidfVectorizer, split=split, paragraph=True, no_vec=no_vec)
def BOW_matched_paragraph_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, CountVectorizer, split=split, paragraph=True, no_vec=no_vec)

def TFIDF_matched_sentence_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, TfidfVectorizer, split=split, sentence=True, no_vec=no_vec)
def BOW_matched_sentence_dataset(split=True, no_vec=False):
    return Dataset(matched_negatives, matched_positives, CountVectorizer, split=split, sentence=True, no_vec=no_vec)

if __name__ == "__main__":
    TFIDF_matched_dataset(split=False, no_vec=True).output("./TFIDF_matched_dataset.csv")