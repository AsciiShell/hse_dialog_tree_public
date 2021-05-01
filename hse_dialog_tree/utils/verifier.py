import hashlib
import pickle

# Protocol 4 can produce different results
HIGHEST_PROTOCOL = 3


def get_hash(obj, protocol: int = HIGHEST_PROTOCOL):
    return hashlib.sha256(pickle.dumps(obj, protocol=protocol)).hexdigest()
