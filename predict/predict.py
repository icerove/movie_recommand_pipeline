from surprise import SVD
from surprise import Dataset
from surprise.dump import dump, load
import os

# Load model
_, algo = load("/pfs/train/model")

# Load each new committed data to make prediction
for dirpath, dirs, files in os.walk("/pfs/streaming"):
    for file in files:
        filepath = os.path.join(dirpath, file)
        # Save prediction result to same filename in pipeline output repo
        outputpath = os.path.join("/pfs/out/", file)
        with open(filepath) as f:
            with open (outputpath, 'w') as o:
                for line in f.readlines():
                    uid, iid = line.strip().split(" ")
                    pred = algo.predict(uid, iid)
                    o.write(uid + " " + iid + " " + str(pred.est) + "\n")
