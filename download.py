import os
import hashlib
import tarfile
import zipfile
import requests

#@save
def download(name, DATA_HUB, cache_dir=os.path.join('..', 'data')):
    """download dataset, and return the file name"""
    assert name in DATA_HUB, f"{name} doesn't have {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    if os.path.exists(fname):   # if file already exists, stop download and return
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)   # calculate sha1
        if sha1.hexdigest() == sha1_hash:
            return fname
        
    """download"""
    print(f'downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, DATA_HUB, folder=None):
    """download and extract dataset"""

    """download"""
    fname = download(name, DATA_HUB)

    base_dir = os.path.dirname(fname)  
    if os.path.exists(os.path.join(base_dir, name)):   # if file has already been extracted, stop and return
        print(f'{name} already exists.')
        return os.path.join(base_dir, name)
    
    """extract"""
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'only zip/tar file can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_imdb():
    """download IMDb dataset"""
    DATA_HUB = dict()
    DATA_HUB['aclImdb'] = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', '01ada507287d82875905620988597833ad4e0903')
    data_dir = download_extract('aclImdb', DATA_HUB, 'aclImdb')
    return data_dir




