from dataset import TFIDF_matched_dataset, TFIDF_matched_paragraph_dataset, TFIDF_matched_sentence_dataset
import numpy as np
import coremltools as ct
from glob import glob
from tqdm import tqdm

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
model_paths = glob("./BotOrNot.mlproj/Models/*.mlmodel")

class CoreMLClassifier(object):
    def __init__(self, model_path, labels=["Human", "ChatGPT"]):
        self.path = model_path
        self.model = ct.models.MLModel(model_path)
        self.labels = labels
    
    def evaluate(self, dataset, verbose=True):
        self.positive_predicted = self.predict(dataset.positive_texts)
        self.negative_predicted = self.predict(dataset.negative_texts)
        self.positive_acc = self.positive_predicted.count(1)/len(self.positive_predicted)
        self.negative_acc = self.negative_predicted.count(0)/len(self.negative_predicted)
        self.combined_acc = (self.positive_predicted.count(1) + self.negative_predicted.count(0))/(len(self.positive_predicted) + len(self.negative_predicted))
        for i in ["Positive Accuracy:", str(self.positive_acc), "Negative Accuracy:", self.negative_acc, "Combined Accuracy:", self.combined_acc] if verbose else []:
            print(i)
    
    def predict(self, text):
        if type(text) in [str]:
            return int(self.model.predict({"text": text})["label"])
        else:
            ret = []
            for t in tqdm(text):
                try:
                    ret.append(int(self.model.predict({"text": t})["label"]))
                except Exception as err:
                    pass
            return ret
    
    def classify(self, text):
        return np.array(self.labels)[self.predict(text)]

if __name__ == "__main__":
    models = [CoreMLClassifier(i) for i in model_paths]
    print("FULL")
    ds = TFIDF_matched_dataset(split=False, no_vec=True)
    for model in models:
        print(model.path.split("/")[-1].split(".")[0] + ":")
        model.evaluate(ds)
    del ds
    print("PARAGRAPH")
    pds = TFIDF_matched_paragraph_dataset(split=False, no_vec=True)
    for model in models:
        print(model.path.split("/")[-1].split(".")[0] + ":")
        model.evaluate(pds)
    del pds
    print("SENTENCE")
    pds = TFIDF_matched_sentence_dataset(split=False, no_vec=True)
    for model in models:
        print(model.path.split("/")[-1].split(".")[0] + ":")
        model.evaluate(pds)
    del pds