RESUME = True
ERR_TEXT = "Unusable response produced by ChatGPT, maybe its unavailable."
MODE = ["examples", "equalized_length"][1]
LOG_FILE = "./log.txt"

_print = print

def print(*args, **kwargs):
    _print(*args, **kwargs)
    with open(LOG_FILE, "a+") as doc:
        for i in args:
            doc.write(str(i) + "\n")

from time import sleep
print("sleeping 15")
sleep(15)

from os.path import isfile
from glob import glob
from tqdm import tqdm
from chatgpt_wrapper import ChatGPT
from random import sample
import subprocess
from os import system

err_text = False
err_err = False

print("starting")

try:
    bot = ChatGPT()
    print("bot started")
except Exception as eerr:
    try:
        bot.page.close()
    except Exception as err:
        print(type(err), err)
    try:
        bot.browser.close()
    except Exception as err:
        print(type(err), err)
    try:
        bot.play.stop()
    except Exception as err:
        print(type(err), err)
    try:
        del bot
    except Exception as err:
        print(type(err), err)
    bot = None
    print(type(eerr), eerr, "bot start failed sleeping 20")
    err_err = True
    sleep(20)

if not err_err:
    _unfinished = glob("./data/positives/{}/*.txt".format(MODE))
    unfinished = []
    for i in _unfinished:
        with open(i, "r") as doc:
            if "." not in doc.read()[-5:]:
                unfinished.append(i)

    try:
        print("started")
        for i in tqdm(unfinished):
            if isfile("./die.die"):
                system("rm ./die.die")
                print("***DYING***")
                exit()
            prompt = ""
            if MODE == "examples":
                with open(i, "r") as doc:
                    prompt = "finish the following essay: " + doc.read()
            elif MODE == "equalized_length":
                with open("./data/negatives/examples/{}.txt".format(i.split("/")[-1].split(".")[0]), "r") as doc:
                    paragraph_len = len(doc.read().split("\n"))
                with open(i, "r") as doc:
                    prompt = "finish the following " + str(paragraph_len) + " paragraph essay: " + doc.read()
            response = bot.ask(prompt)
            if ERR_TEXT not in response:
                with open(i, "a+") as doc:
                    doc.write(response.replace("\n\n", "\n"))
            else:
                print("ERR_TEXT")
                err_text = True
                break
    except Exception as err:
        error = err
        err_err = True
        print(type(err), err)

if err_text or err_err:
    try:
        bot.page.close()
    except Exception as err:
        print(type(err), err)
    try:
        bot.browser.close()
    except Exception as err:
        print(type(err), err)
    try:
        bot.play.stop()
    except Exception as err:
        print(type(err), err)
    try:
        del bot
    except Exception as err:
        print(type(err), err)
    sleep(10)
    subprocess.Popen(["python3", "./cutoff_restore.py"])
print("exiting")