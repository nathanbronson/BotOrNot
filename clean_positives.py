from glob import glob
from os import system

DIR = ["examples", "equalized_length"][1]

to = []
ts = []
eoff = []
files = glob("./data/positives/{}/*.txt".format(DIR))

for i in files:
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
        with open(i, "w+") as doc:
            doc.write(r.replace("\n\n", "\n"))
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
if len(to) > 0:
    print("\t" + "\n\t\t".join(to))
print("\tShorts: " + str(len(ts)))
if len(ts) > 0:
    print("\t" + "\n\t\t".join(ts))
print("\tEOF Flags: " + str(len(eoff)))
if len(eoff) > 0:
    print("\t" + "\n\t\t".join(eoff))