from surprise import SVD
from surprise import Dataset, Reader
from surprise.dump import dump, load
import os

if os.path.exists("/pfs/out/model"):
    # If we have model saved by previous training, load it
    _, algo = load("/pfs/out/model")
    reader = Reader(line_format='user item rating timestamp', sep=' ')

    # Train model with each new committed train data
    for dirpath, dirs, files in os.walk("/pfs/training"):
        for filename in files:
            filepath = os.path.join(dirpath, file)
            with open(filepath) as f:
                data = Dataset.load_from_file(filepath, reader=reader).build_full_trainset()
                algo.fit(data)
else:
    # If it's initial run, train with existing dataset

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k').build_full_trainset()

    # We'll use the famous SVD algorithm.
    algo = SVD()
    algo.fit(data)

# In both case, save trained model
dump("/pfs/out/model", algo=algo)