import random

# 6种蛋白质序列数据增强方案复现

def replacement_dict(seq: str, p: float):
    """
    基于字典随机替换RD
    """
    dicts = {"A": "V", "V": "A",
             "S": "T", "T": "S",
             "F": "Y", "Y": "F",
             "K": "R", "R": "K",
             "C": "M", "M": "C",
             "D": "E", "E": "D",
             "N": "Q", "Q": "N",
             "V": "I", "I": "V"
             }
    argumented_seq = []
    for a in seq:
        if a in dicts.keys() and random.random() <= p:
            argumented_seq.append(dicts[a])
        else:
            argumented_seq.append(a)
    return "".join(argumented_seq)

def replacement_alanine(seq: str, p: float):
    """
    随机替换成alanine(A), RA
    """
    argumented_seq = []
    for a in seq:
        if random.random() <= p:
            argumented_seq.append('A')
        else:
            argumented_seq.append(a)
    return "".join(argumented_seq)

def global_random_shuffling(seq: str):
    """
    全局打乱氨基酸位置
    """
    seq_list = [x for x in seq]
    random.shuffle(seq_list)
    return "".join(seq_list)

def local_random_shuffling(seq: str):
    """
    在某个范围内[l, r]打乱氨基酸位置
    """
    n = len(seq)
    l = random.randint(0, n - 1)
    r = random.randint(l, n - 1)
    seq_list = [x for x in seq]
    sub_seq = seq_list[l : r + 1]
    random.shuffle(sub_seq)
    seq_list[l: r + 1] = sub_seq

    return "".join(seq_list)

def sequence_revsersion(seq: str):
    """
    在某个范围内[l, r]反转氨基酸序列
    """
    n = len(seq)
    l = random.randint(0, n - 1)
    r = random.randint(l, n - 1)
    seq_list = [x for x in seq]
    sub_seq = seq_list[l : r + 1]
    sub_seq.reverse()
    seq_list[l: r + 1] = sub_seq

    return "".join(seq_list)

def sequence_subsampling(seq: str):
    """
    在某个范围内[l, r]选取子序列
    要求子序列长度至少为10
    """
    n = len(seq)
    if n <= 10: return seq
    l = random.randint(0, n - 1 - 10)
    r = random.randint(l + 10 - 1 , n - 1)
    seq_list = [x for x in seq]
    sub_seq = seq_list[l : r + 1]

    return "".join(sub_seq)

# if __name__ == '__main__':
#     s = "FDVMGIIKKIAGAL"
#     # r = replacement_alanine("FDVMGIIKKIAGAL", 0.01)
#     for i in range(100):
#         r = sequence_subsampling("FDVMGIIKKIAGAL")
