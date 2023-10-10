import os
from text_splitter import split_by_heading
from tqdm import tqdm

def unicode2str(filename):
    u = filename[:-len(".txt")]
    code_points = u.split("#")[1:]
    s = ""
    for cp in code_points:
        s += chr(int(cp[1:5], 16))
        s += cp[5:] # for trailing numbers not starting with #U
    return s

def load_data(filenames, split=True):
    doc_list = []
    for filename in tqdm(filenames):
        with open(filename, "r") as f:
            title = unicode2str(os.path.basename(filename))
            if not split:
                data = f.read()
                doc_list.append({
                    "标题": title,
                    "子标题": "",
                    "内容": data
                })
            else:
                data = split_by_heading(filename)
                for key, val in data.items():
                    doc_list.append({
                        "标题": title,
                        "子标题": key,
                        "内容": val
                    })
    return doc_list

def load_data_from_dir(dir, split=True):
    filenames = [os.path.join(dir, filename) for filename in os.listdir(dir)]
    return load_data(filenames, split=split)