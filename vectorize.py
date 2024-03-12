from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def TFIDF(x):
    vectorizer = TfidfVectorizer()
    transform = vectorizer.fit_transform(x)
    return transform, vectorizer

def BOW(x):
    vectorizer = CountVectorizer()
    transform = vectorizer.fit_transform(x)
    return transform, vectorizer