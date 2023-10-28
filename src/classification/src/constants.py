LABEL2ID = {
    "first_level": {
        "policy": 0,
        "knowledge_base": 1,
        "chat": 2,
        "policy_tag": 3,
        "others": 4,
    },
    "second_level": {
        "policy": {
            "basic_info": 0,
            "award": 1,
            "process": 2,
            "materials": 3,
            "condition": 4
        }
    },
    "direct": { # only consider policy and knowledge_base
        "basic_info": 0,
        "award": 1,
        "process": 2,
        "materials": 3,
        "condition": 4,
        "knowledge_base": 5,
        "others": 6,
    }
}

def get_id2label(label2id):
    tmp = {}
    for k, v in label2id.items():
        if not isinstance(v, dict):
            tmp[v] = k
        else:
            tmp[k] = get_id2label(v)
    return tmp

ID2LABEL = get_id2label(LABEL2ID)
