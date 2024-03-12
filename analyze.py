from classify import p_TFIDF_matched_bayes, p_TFIDF_matched_rf, p_TFIDF_matched_sgd, p_TFIDF_matched_svc
from glob import glob
from random import sample

NUMBER_OF_FILES = 10

files = sample(glob("./data/external_data/negatives/*.txt") + glob("./data/external_data/positives/*.txt"), NUMBER_OF_FILES)
texts = []
for i in files:
    with open(i, "r") as doc:
        texts.append(doc.read())

models = ["p_TFIDF_matched_bayes", "p_TFIDF_matched_rf"]#, "p_TFIDF_matched_sgd", "p_TFIDF_matched_svc"] #sgd/svc incompatible
for i in models:
    eval(i).train(verbose=False)

if __name__ == "__main__":
    for i in models:
        eval(i).lime("./lime_visualizations/lime_{}.html".format(i + "_{}"), text_instances=len(texts), data_source=texts)