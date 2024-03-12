from pickle import load
from lime.lime_text import LimeTextExplainer
import numpy as np
from textract import process
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

CLASSES = ["Human", "ChatGPT"]
ESSAY_FORM = {
    "file_path": None,
    "name": None,
    "text": None,
    "html_report": None,
    "predicted_class": None
}
CPU = cpu_count()/2 - 2
CPU = 1 if CPU < 1 else CPU

with open("/".join(__file__.split("/")[:-1]) + "/models/model.pkl", "rb") as doc:
    classifier = load(doc)
lte = LimeTextExplainer(class_names=["Human", "ChatGPT"])

def lime(text, num_features=50):
    """
    Returns HTML and predicted class for lime visiualization of text cassification
    """
    assert type(text) is str
    return lte.explain_instance(text, classifier.predict_proba, num_features=num_features)

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
            "name": file.split("/")[-1].split(".")[0].replace("-", " ").replace("_", " "),
            "text": texts[file],
            "html_report": reports[file].as_html(),
            "predicted_class": CLASSES[np.argmax(reports[file].predict_proba)],
            "human_probability": reports[file].predict_proba[0],
            "chatgpt_probability": reports[file].predict_proba[1],
            "full_report": reports[file]
        })
    return essays

def files_to_essays_gen(files, num_features=50):
    """
    Turn list of filepaths to essay form (main backend function)
    """
    for file in files:
        try:
            if file.split(".")[-1] in ["txt", "text"]:
                with open(file, "r", encoding="utf-8") as doc:
                    text = doc.read()
            else:
                text = process(file).decode("utf-8")
        except Exception as err:
            try:
                with open(file, "r", encoding="ascii") as doc:
                    text = doc.read()
            except Exception as err:
                print("OPENING ERROR:", type(err), err)
                continue
        l = lime(text, num_features=num_features)
        yield {
            "file_path": file,
            "name": file.split("/")[-1].split(".")[0].replace("-", " ").replace("_", " "),
            "text": text,
            "html_report": l.as_html(),
            "predicted_class": CLASSES[np.argmax(l.predict_proba)],
            "human_probability": l.predict_proba[0],
            "chatgpt_probability": l.predict_proba[1],
            "full_report": l
        }

def files_to_essays_gen_multi(files, num_features=50):
    """
    Turn list of filepaths to essay form (main backend function)
    """
    with ProcessPoolExecutor(max_workers=CPU) as executor:
        procs = []
        for file in files:
            try:
                if file.split(".")[-1] in ["txt", "text"]:
                    with open(file, "r", encoding="utf-8") as doc:
                        text = doc.read()
                else:
                    text = process(file).decode("utf-8")
            except Exception as err:
                try:
                    with open(file, "r", encoding="ascii") as doc:
                        text = doc.read()
                except Exception as err:
                    print("OPENING ERROR:", type(err), err)
                    continue
            procs.append((file, text, executor.submit(lime, text, num_features=num_features)))
        for file, text, _l in procs:
            l = _l.result()
            yield {
                "file_path": file,
                "name": file.split("/")[-1].split(".")[0].replace("-", " ").replace("_", " "),
                "text": text,
                "html_report": l.as_html(),
                "predicted_class": CLASSES[np.argmax(l.predict_proba)],
                "human_probability": l.predict_proba[0],
                "chatgpt_probability": l.predict_proba[1],
                "full_report": l
            }

def text_to_essay(text, num_features=50):
    """
    Turn Text to Essay
    """
    assert type(text) in [str], type(text)
    l = lime(text, num_features=num_features)
    return {
        "file_path": "",
        "name": text,
        "text": text,
        "html_report": l.as_html(),
        "predicted_class": CLASSES[np.argmax(l.predict_proba)],
        "human_probability": l.predict_proba[0],
        "chatgpt_probability": l.predict_proba[1],
        "full_report": l
    }