import re

chinese_num = "零一二三四五六七八九十"
pattern_list1 = [
    r"^[%s]\s*[、\.]" % chinese_num, # 一、 一.
    # r"^[（〔\(][%s][〕）\)]\s*" % chinese_num, #（一）〔一〕(一) 
    # r"^第\s*[%s\d]+\s*部分\s*" % chinese_num, # 第一部分
    # r"^\d+\s*[、\)〕）]\s*", # 1、1)
    # r"^[（\(〔]\d+[）\)〕]", # （1）(1) 〔1〕
    # r"^\d+(\.\d+)+\s*", # 1.2.3
    # r"^\d+(-\d+)*\s*", # 1-1, 1-1-1, 1-1-1-1 ... 
    r"^第\s*[%s\d]+\s*章\s*" % chinese_num, # 第一章
    r"^第\s*[%s\d]+\s*条\s*" % chinese_num, # 第一条
    # r"^_*\d+\s*[．\.]\s+", # 1．1. 
]
start_pattern1 = re.compile("|".join(pattern_list1))

pattern_list2 = [
    r"^[%s]\s*[、\.]" % chinese_num, # 一、 一.
    r"^[（〔\(][%s][〕）\)]\s*" % chinese_num, #（一）〔一〕(一)
    # r"^第\s*[%s\d]+\s*部分\s*" % chinese_num, # 第一部分
    # r"^\d+\s*[、\)〕）]\s*", # 1、1)
    # r"^[（\(〔]\d+[）\)〕]", # （1）(1) 〔1〕
    # r"^\d+(\.\d+)+\s*", # 1.2.3
    # r"^\d+(-\d+)*\s*", # 1-1, 1-1-1, 1-1-1-1 ... 
    r"^第\s*[%s\d]+\s*章\s*" % chinese_num, # 第一章
    r"^第\s*[%s\d]+\s*条\s*" % chinese_num, # 第一条
    # r"^_*\d+\s*[．\.]\s+", # 1．1. 
]
start_pattern2 = re.compile("|".join(pattern_list2))

chinese_punctuations = "，。、；：？！…—·「」『』（）［］【】《》〈〉“”‘’. "
english_punctuations = ",.;:?!…—·\"'()[]{}<>"

non_printable_characters = [
    "\u200b", # zero width space
    "\u200e", # left-to-right mark
    "\u200f", # right-to-left mark
    "\u202a", # left-to-right embedding
    "\u202b", # right-to-left embedding
    "\u202c", # pop directional formatting
    "\u202d", # left-to-right override
    "\u202e", # right-to-left override
    "\u2060", # word joiner
    "\u2061", # function application
    "\u2062", # invisible times
    "\u2063", # invisible separator
    "\u2064", # invisible plus
    "\u2066", # left-to-right isolate
    "\u2067", # right-to-left isolate
    "\u2068", # first strong isolate
    "\u2069", # pop directional isolate
    "\u206a", # inhibit symmetric swapping
    "\u206b", # activate symmetric swapping
    "\u206c", # inhibit Arabic form shaping
    "\u206d", # activate Arabic form shaping
    "\u206e", # national digit shapes
    "\u206f", # nominal digit shapes
    "\ufeff", # zero width no-break space
    "\ufff9", # interlinear annotation anchor
    "\ufffa", # interlinear annotation separator
    "\ufffb", # interlinear annotation terminator
]

def postprocess(text):
    text = text.replace(" ", "")
    text = text.replace("　", "")
    # text = text.strip("\n "+chinese_punctuations+english_punctuations)
    # replace invisible characters
    for ch in non_printable_characters:
        text = text.replace(ch, "")
    # # convert to simplified chinese
    # text = converter.convert(text)
    return text


def match_and_split_heading1(text, pattern=start_pattern1):
    text = text.strip()
    match = pattern.match(text)
    if match:
        heading_rank = match.group().strip()
        heading_text = postprocess(text[match.end():])
        return heading_rank, heading_text
    else:
        return None, postprocess(text)


def match_and_split_heading2(text, pattern=start_pattern2):
    text = text.strip()
    match = pattern.match(text)
    if match:
        heading_rank = match.group().strip()
        heading_text = postprocess(text[match.end():])
        return heading_rank, heading_text
    else:
        return None, postprocess(text)



def get_heading_info1(text):
    for idx in range(len(pattern_list1)):
        heading_rank, heading_text = match_and_split_heading1(text, 
                                    re.compile(pattern_list1[idx]))
        if heading_rank:
            return idx, heading_rank, heading_text
    return None, None, text

def get_heading_info2(text):
    for idx in range(len(pattern_list2)):
        heading_rank, heading_text = match_and_split_heading2(text, 
                                    re.compile(pattern_list2[idx]))
        if heading_rank:
            return idx, heading_rank, heading_text
    return None, None, text



def split_txt_by_heading(contents, min_sentence_len=2, max_heading_len=200):
    if "(1)" not in "".join(contents) and "（1）" not in "".join(contents):
        doc_tree = {}
        heading_stack = []
        # get policy introduction
        intro = ""
        find_heading = False
        heading_stack_=[]
        key_list=[]
        for line in contents:
            line = line.strip("_\n ")
            line = line.replace(u"\xa0", " ")
            if len(line) < min_sentence_len:
                continue
            heading_type, heading_rank, heading_text = get_heading_info1(line)
            if heading_type is not None:
                find_heading = True
            if not find_heading:
                intro += line + "\n"
            heading_token_num = len(heading_text)
            if heading_type is not None: # and heading_token_num < max_heading_len:
                for idx in range(len(heading_stack)):
                    if heading_stack[idx][0] == heading_type:
                        heading_stack = heading_stack[:idx]
                        break
                heading_stack.append((heading_type, heading_rank, heading_text))
                if heading_stack not in heading_stack_:
                    heading_stack_.append(heading_stack)
            else:
                if len(heading_stack) > 0:
                    key = " ".join([x[1] + x[2] if x[1][-1]=="、" else x[1] + "、" + x[2]
                                    for x in heading_stack])
                    if key not in doc_tree:
                        doc_tree[key] = ""
                        key_list.append(key)
                    doc_tree[key] += postprocess(line)



        for j in range(0,len(heading_stack_)):
            # print("heading_stack_", heading_stack_[j])
            key = " ".join([x[1] + x[2] if x[1][-1] == "、" else x[1] + "、" + x[2]
                            for x in heading_stack_[j]])
            if key not in key_list:
                doc_tree[key] = ""

        if len(intro) > 0:
            doc_tree["简介"] = intro
        return doc_tree
    else:
        doc_tree = {}
        heading_stack = []
        # get policy introduction
        intro = ""
        find_heading = False
        heading_stack_ = []
        key_list = []
        for line in contents:
            line = line.strip("_\n ")
            line = line.replace(u"\xa0", " ")
            if len(line) < min_sentence_len:
                continue
            heading_type, heading_rank, heading_text = get_heading_info2(line)
            if heading_type is not None:
                find_heading = True
            if not find_heading:
                intro += line + "\n"
            heading_token_num = len(heading_text)
            if heading_type is not None:  # and heading_token_num < max_heading_len:
                for idx in range(len(heading_stack)):
                    if heading_stack[idx][0] == heading_type:
                        heading_stack = heading_stack[:idx]
                        break
                heading_stack.append((heading_type, heading_rank, heading_text))
                if heading_stack not in heading_stack_:
                    heading_stack_.append(heading_stack)
            else:
                if len(heading_stack) > 0:
                    key = " ".join([x[1] + x[2] if x[1][-1] == "、" else x[1] + "、" + x[2]
                                    for x in heading_stack])
                    if key not in doc_tree:
                        doc_tree[key] = ""
                        key_list.append(key)
                    doc_tree[key] += postprocess(line)

        for j in range(0, len(heading_stack_)):
            # print("heading_stack_", heading_stack_[j])
            key = " ".join([x[1] + x[2] if x[1][-1] == "、" else x[1] + "、" + x[2]
                            for x in heading_stack_[j]])
            if key not in key_list:
                doc_tree[key] = ""

        if len(intro) > 0:
            doc_tree["简介"] = intro
        return doc_tree




def split_by_heading(filename, min_sentence_len=2):
    file_format = filename.split(".")[-1]
    assert file_format == "txt"
    with open(filename, "r", encoding="utf-8") as fin:
        contents = fin.readlines()
    return split_txt_by_heading(contents, min_sentence_len=min_sentence_len)
