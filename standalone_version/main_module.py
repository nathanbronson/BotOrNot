from pickle import load
from lime.lime_text import LimeTextExplainer
import numpy as np

CLASSES = ["Human", "ChatGPT"]
ESSAY_FORM = {
    "file_path": None,
    "name": None,
    "text": None,
    "html_report": None,
    "predicted_class": None
}

with open("./models/model.pkl", "rb") as doc:
    classifier = load(doc)
lte = LimeTextExplainer(class_names=["Human", "ChatGPT"])

def lime(text, num_features=50):
    """
    Returns HTML and predicted class for lime visiualization of text cassification
    """
    assert type(text) is str
    #ei = lte.explain_instance(text, classifier.predict_proba, num_features=num_features)
    return lte.explain_instance(text, classifier.predict_proba, num_features=num_features)
    return (ei.as_html(), CLASSES[np.argmax(ei.predict_proba)])

def create_reports(files, num_features=50):
    """
    Create lime reports for a set of files
    """
    texts = {}
    for file in files:
        with open(file, "r") as doc:
            texts[file] = doc.read()
    return {file: lime(text, num_features=num_features) for (file, text) in texts.items()}, texts

def files_to_essays(files, num_features=50):
    """
    Turn list of filepaths to essay form (main backend function)
    """
    essays = []
    reports, texts = create_reports(files, num_features=num_features)
    for file in files:
        essays.append({
            "file_path": file,
            "name": file.split("/")[-1].split(".")[0],
            "text": texts[file],
            "html_report": reports[file].as_html(),
            "predicted_class": CLASSES[np.argmax(reports[file].predict_proba)],
            "human_probability": reports[file].predict_proba[0],
            "chatgpt_probability": reports[file].predict_proba[1],
            "full_report": reports[file]
        })
    return essays

if __name__ == "__main__":
    e = files_to_essays(["../data/external_data/negatives/cnn_article.txt", "../data/external_data/positives/manifesto_with_and__without.txt"])
    print(e[0]["predicted_class"], e[1]["predicted_class"])
    print(e[0]["human_probability"], e[1]["human_probability"])
    print(e[0]["chatgpt_probability"], e[1]["chatgpt_probability"])