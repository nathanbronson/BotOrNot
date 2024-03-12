from dataset import TFIDF_matched_dataset, BOW_matched_dataset, TfidfVectorizer, CountVectorizer, TFIDF_matched_paragraph_dataset, BOW_matched_paragraph_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from pickle import load, dump
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer

SGD_KWARGS = {
    "loss": "hinge",
    "penalty": "l2",
    "alpha": 1e-3,
    "max_iter": 100,
    "tol": None
}
SVC_KWARGS = {
    "gamma": "auto",
    "max_iter": 100
}
SPLIT = True
PARAGRAPH = False

class PipelineClassifier(object):
    def __init__(self, dataset, model, vectorizer, classifier_args=(), classifier_kwargs={}, vectorizer_args=(), vectorizer_kwargs={}):
        self.dataset = dataset
        self.pipeline = Pipeline([
            ("vec", vectorizer(*vectorizer_args, **vectorizer_kwargs)),
            ("clf", model(*classifier_args, **classifier_kwargs))
        ])
        self.lte = LimeTextExplainer(class_names=["Human", "ChatGPT"])
    
    def train(self, verbose=True):
        self.fit()
        self.evaluate(verbose=verbose)
    
    def fit(self):
        self.pipeline.fit(*self.dataset.text_fit())
    
    def evaluate(self, verbose=True):
        self.test_predicted = self.pipeline.predict(self.dataset.x_test_text)
        self.test_accuracy = np.mean(self.test_predicted == self.dataset.y_test_text)
        self.test_metrics = metrics.classification_report(self.dataset.y_test_text, self.test_predicted)
        self.test_confusion = metrics.confusion_matrix(self.dataset.y_test_text, self.test_predicted)
        for i in ["Test Accuracy:", str(self.test_accuracy), "Test Metrics:", self.test_metrics, "Test Confusion Matrix:", self.test_confusion] if verbose else []:
            print(i)
    
    def predict(self, text):
        if type(text) in [str]:
            return self.pipeline.predict([text])
        else:
            return self.pipeline.predict(text)
    
    def classify(self, text):
        return self.predict(text)
    
    def save(self, path="./models/model.py"):
        with open(path, "wb") as doc:
            dump(self.pipeline, doc)
    
    def load(self, path):
        print("Not Yet Implemented for Pickle Load")
    
    def lime(self, path="./lime_visualizations/lime_{}.html", num_features=50, text_instances=10, data_source=None, _try=True): #hopefully incorporate shap as well
        for i in range(text_instances):
            try:
                self.lte.explain_instance((self.dataset.x_test_text if data_source is None else data_source)[i], self.pipeline.predict_proba, num_features=num_features).save_to_file(path.format(str(i + 1)))
            except Exception as err:
                if _try:
                    print(type(err), err, i)
                else:
                    raise err

class Classifier(object):
    def __init__(self, dataset, model, classifier_kwargs={}):
        self.dataset = dataset
        self.model = model(**classifier_kwargs)
    
    def _vectorize(self, text):
        if type(text) in [str]:
            return self.dataset.vectorizer.transform([text])
        else:
            return self.dataset.vectorizer.transform(text)

    def train(self):
        self.fit()
        self.evaluate()
    
    def fit(self):
        self.model.fit(*self.dataset.vectorized_fit())
    
    def evaluate(self, verbose=True):
        self.test_predicted = self.model.predict(self.dataset.x_test)
        self.test_accuracy = np.mean(self.test_predicted == self.dataset.y_test)
        self.test_metrics = metrics.classification_report(self.dataset.y_test, self.test_predicted)
        self.test_confusion = metrics.confusion_matrix(self.dataset.y_test, self.test_predicted)
        for i in ["Test Accuracy:", str(self.test_accuracy), "Test Metrics:", self.test_metrics, "Test Confusion Matrix:", self.test_confusion] if verbose else []:
            print(i)
    
    def predict(self, text):
        return self.model.predict(self._vectorize(text))
    
    def classify(self, text):
        return self.predict(text)
    
    def save(self, path):
        print("Not Yet Implemented for Pickle Save")
    
    def load(self, path):
        print("Not Yet Implemented for Pickle Load")

if PARAGRAPH:
    ds = TFIDF_matched_paragraph_dataset(split=SPLIT)
    bds = BOW_matched_paragraph_dataset(split=SPLIT)
else:
    ds = TFIDF_matched_dataset(split=SPLIT)
    bds = BOW_matched_dataset(split=SPLIT)

TFIDF_matched_bayes = Classifier(ds, MultinomialNB)
TFIDF_matched_sgd = Classifier(ds, SGDClassifier, classifier_kwargs=SGD_KWARGS)
TFIDF_matched_svc = Classifier(ds, SVC, classifier_kwargs=SVC_KWARGS)
TFIDF_matched_rf = Classifier(ds, RandomForestClassifier)

p_TFIDF_matched_bayes = PipelineClassifier(ds, MultinomialNB, TfidfVectorizer)
p_TFIDF_matched_sgd = PipelineClassifier(ds, SGDClassifier, TfidfVectorizer, classifier_kwargs=SGD_KWARGS)
p_TFIDF_matched_svc = PipelineClassifier(ds, SVC, TfidfVectorizer, classifier_kwargs=SVC_KWARGS)
p_TFIDF_matched_rf = PipelineClassifier(ds, RandomForestClassifier, TfidfVectorizer)

p_BOW_matched_bayes = PipelineClassifier(bds, MultinomialNB, CountVectorizer)
p_BOW_matched_sgd = PipelineClassifier(bds, SGDClassifier, CountVectorizer, classifier_kwargs=SGD_KWARGS)
p_BOW_matched_svc = PipelineClassifier(bds, SVC, CountVectorizer, classifier_kwargs=SVC_KWARGS)
p_BOW_matched_rf = PipelineClassifier(bds, RandomForestClassifier, CountVectorizer)

if __name__ == "__main__":
    print("TFIDF Matched Bayes")
    p_TFIDF_matched_bayes.train()
    print("TFIDF Matched SGD")
    p_TFIDF_matched_sgd.train()
    print("TFIDF Matched SVC")
    p_TFIDF_matched_svc.train()
    print("TFIDF Matched RF")
    p_TFIDF_matched_rf.train()
    print("ELI5")
    p_TFIDF_matched_rf.lime()
    print("BOW Matched Bayes")
    p_BOW_matched_bayes.train()
    print("BOW Matched SGD")
    p_BOW_matched_sgd.train()
    print("BOW Matched SVC")
    p_BOW_matched_svc.train()
    print("BOW Matched RF")
    p_BOW_matched_rf.train()
    print("ELI5")
    p_BOW_matched_rf.lime()
    print("Saving TFIDF Random Forest")
    p_TFIDF_matched_rf.save("./models/TFIDF_matched_Random_Forest_pipeline.pkl")