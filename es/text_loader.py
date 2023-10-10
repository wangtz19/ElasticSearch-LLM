import os
from tqdm import tqdm

def unicode2str(filename):
    u = filename[:-len(".txt")]
    code_points = u.split("#")[1:]
    s = ""
    for cp in code_points:
        s += chr(int(cp[1:5], 16))
        s += cp[5:] # for trailing numbers not starting with #U
    return s

def load_data(filenames):
    doc_list = []
    for filename in tqdm(filenames):
        with open(filename, "r") as f:
            data = f.read()
            title = unicode2str(os.path.basename(filename))
            doc_list.append({
                "标题": title,
                "内容": data
            })
    return doc_list

def load_data_from_dir(dir):
    filenames = [os.path.join(dir, filename) for filename in os.listdir(dir)]
    return load_data(filenames)