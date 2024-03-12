from glob import glob
from os import system
import re
from copy import copy
from tqdm import tqdm

UNACCEPTABLE_PREFIXES = []
LETTERS = [i for i in "abcdefghijklmnopqrstuvwxyz"]
UNACCEPTABLE_SUFFIXES = []
PRINT_NAMES = False

cite1 = re.compile(r"[A-Z][a-zA-Z]*?, [A-Z][a-zA-Z]*?[\.]")
cite2 = re.compile(r" [A-Z][a-zA-Z]+?, \(?[0-9]{2,4}\)?\.")#re.compile(r" (?:[A-Z][a-zA-Z]+? ?)+?, \(?[0-9]{2,4}\)?\.")
cite3 = re.compile(r"\. \(?[0-9]*?\)?\.")
cite4 = re.compile(r", vol\. [0-9]+?[,\.]")

to = []
ts = []
eoff = []
pref = []
suf = []
lets = []
par = []
cit = []
files = glob("./data/negatives/examples/*.txt")

for i in tqdm(files):
    delete = False
    with open(i, "r+") as doc:
        r = doc.read()
    if len(r.split(" ")) < 50 or len(r.split("\n")) < 2:
        ts.append(i)
        delete = True
    elif "\n\nxXtimeoutbrokenXx" in r:
        to.append(i)
        delete = True
    else:
        lines = r.replace("\n\n", "\n").split("\n")

        o_len = len(lines)
        #print(o_len)
        lines = list(filter(lambda e: not any(e.startswith(n) or e.endswith(n) for n in UNACCEPTABLE_PREFIXES), lines)) #remove lines that start or end with a bad prefix
        if o_len > len(lines):
            pref.append(i)
        
        o_len = len(lines)
        #print(o_len)
        lines = list(filter(lambda e: not any(e.startswith(n) or e.endswith(n) for n in UNACCEPTABLE_SUFFIXES), lines)) #remove lines that start or end with a bad suffix
        if o_len > len(lines):
            suf.append(i)
        
        o_len = len(lines)
        #print(o_len)
        lines = list(filter(lambda e: not any(e.startswith(n) or e.endswith(n) for n in LETTERS), lines)) #remove lines that start or end with a lowercase letter
        if o_len > len(lines):
            lets.append(i)
        
        o_lines = copy(lines)
        lines = [re.sub(r"[\(\[].*?[\)\]]", r"", n) for n in lines] #remove parenthetical
        for n in lines:
            if n not in o_lines:
                par.append(i)
                break
        
        o_len = len(lines)
        #print(o_len)
        lines = list(filter(lambda e: not bool(cite1.search(e.split(".")[0])), lines)) #remove some cites
        lines = list(filter(lambda e: not bool(cite2.search(e)), lines))
        lines = list(filter(lambda e: not bool(cite3.search(e)), lines))
        lines = list(filter(lambda e: not bool(cite4.search(e)), lines))
        if o_len > len(lines):
            cit.append(i)
        
        with open(i, "w+") as doc:
            doc.write("\n".join(lines))
    if delete:
        system("rm {}".format(i))
    elif "." not in r[-5:]:
        eoff.append(i)


to = [i.split("/")[-1] for i in to]
ts = [i.split("/")[-1] for i in ts]
eoff = [i.split("/")[-1] for i in eoff]

print("Summary:")
print("\t" + str(len(files)) + " total files")
print("\t" + str(len(to + ts)) + " files removed")
print("\tTimeouts: " + str(len(to)))
if len(to) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(to))
print("\tShorts: " + str(len(ts)))
if len(ts) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(ts))
print("\tEOF Flags: " + str(len(eoff)))
if len(eoff) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(eoff))
print("\tPrefix Flags: " + str(len(pref)))
if len(pref) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(pref))
print("\tSuffix Flags: " + str(len(suf)))
if len(suf) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(suf))
print("\tLetter Flags: " + str(len(lets)))
if len(lets) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(lets))
print("\tParenthetical Flags: " + str(len(par)))
if len(par) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(par))
print("\tCitation Flags: " + str(len(cit)))
if len(cit) > 0 and PRINT_NAMES:
    print("\t" + "\n\t\t".join(cit))