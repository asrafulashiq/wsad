import numpy as np


def str2ind(categoryname, classlist):
    return [
        i for i in range(len(classlist))
        if categoryname == classlist[i].decode("utf-8")
    ][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)],
                  axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_extract(feat, t_max):
    # ind = np.arange(feat.shape[0])
    # splits = np.array_split(ind, t_max)
    # nind = np.array([np.random.choice(split, 1)[0] for split in splits])
    # return feat[nind]

    # ind = np.random.choice(feat.shape[0], size=t_max)
    # ind = sorted(ind)
    # return feat[ind]
    r = np.random.randint(len(feat) - t_max)
    return feat[r: r + t_max]


def pad(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(
            feat,
            ((0, min_len - feat.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        return feat


def fn_normalize(x):
    return (x - np.mean(x, 0, keepdims=True)) / \
            (np.std(x, 0, keepdims=True)+1e-10)

def process_feat(feat, length=None, normalize=False):
    if length is not None:
        if len(feat) > length:
            x = random_extract(feat, length)
        else:
            x = pad(feat, length)
    else:
        x = feat
    if normalize:
        x = fn_normalize(x)
    return x


def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + "-results.log", "a+")
    string_to_write = str(itr)
    if dmap:
        for item in dmap:
            string_to_write += " " + "%.2f" % item
    string_to_write += " " + "%.2f" % cmap
    fid.write(string_to_write + "\n")
    fid.close()
