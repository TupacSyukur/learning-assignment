import pandas as pd
import numpy as np
import random

train = pd.read_excel(
    r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="train")

test = pd.read_excel(
    r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="test")


# test_id = test["id"]
# train_id = train["id"]

# test_x1 = test["x1"]
# test_x2 = test["x2"]
# test_x3 = test["x3"]

# train_x1 = train["x1"]
# train_x2 = train["x2"]
# train_x3 = train["x3"]
# train_y = train["y"]

# result_y = []
# for i in range(len(test_id)):
#     result = []
#     for j in range(len(train_id)):
#         sum = abs(test_x1[i] - train_x1[j]) + abs(test_x2[i] -
#                                                   train_x2[j]) + abs(test_x3[i] - train_x3[j])
#         result.append(sum)
#     res = result.index(min(result))
#     result_y.append(train["y"][res+1])

# print(result_y)

# l = train.iloc[0:10]
# ll = l.loc[0]
# print(train.loc[295])
# print("y =", train["y"][0])
# print(ll)
# print(ll["x1"])
# print(l.loc[10])


# Function knn should be reviewed again!!!
def knn(k, lt, rt, lv, rv, train, v):
    result_y = []
    for i in range(lv, rv, 1):
        result = []
        for j in range(lt, rt, 1):
            sum = np.sqrt(((train["x1"][i] - train["x1"][j]) ** 2) + ((train["x2"]
                          [i] - train["x2"][j]) ** 2) + ((train["x3"][i] - train["x3"][j]) ** 2))
            result.append(sum)
        nearest = []
        for i in range(k):
            idx = result.index(min(result))
            if v == "v1":
                select = train.loc[idx+lt]
            elif v == "v2":
                select = train.loc[idx+rt]
            nearest.append(select["y"])
            result[idx] = np.inf
        if nearest.count(1) > nearest.count(0):
            result_y.append(1)
        elif nearest.count(0) > nearest.count(1):
            result_y.append(0)
        else:
            result_y.append(random.choice([0, 1]))

    return result_y


def validation(k, train):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(1, 9, 1):
        if i == 1:
            lt = 37
            rt = 296
            lv = 0
            rv = 37
            result_y = knn(k, lt, rt, lv, rv, train, "v1")
            for j in range(result_y):
                if result_y[j] == 1:
                    if train.loc["y"][j] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif result_y[j] == 0:
                    if train.loc["y"][j] == 0:
                        tn += 1
                    else:
                        fn += 1
        elif i == 8:
            lt = 0
            rt = 259
            lv = rt
            rv = 296
            result_y = knn(k, lt, rt, lv, rv, train, "v2")
            for j in range(result_y):
                if result_y[j] == 1:
                    if train.loc["y"][j+rt] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif result_y[j] == 0:
                    if train.loc["y"][j+rt] == 0:
                        tn += 1
                    else:
                        fn += 1
        else:
            lt = 0
            rt = (296*(i-1))/8
            lv = rt
            rv = (296*(i))/8
            result_y0 = knn(k, lt, rt, lv, rv, train, "v2")
            lt = (296*(i-1))/8
            rt = 296
            lv = (296*(i-1))/8
            rv = (296*(i))/8
            result_y1 = knn(k, lt, rt, lv, rv, train, "v1")
            result_y = result_y0 + result_y1
            for j in range(result_y):
                if result_y[j] == 1:
                    if train.loc["y"][j+lv] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif result_y[j] == 0:
                    if train.loc["y"][j+lv] == 0:
                        tn += 1
                    else:
                        fn += 1
    confusion_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return confusion_matrix


def performance(conf):
    acc = (conf["tp"] + conf["tn"]) / \
        (conf["tp"] + conf["tn"] + conf["fp"] + conf["fn"])
    recall = conf["tp"] / (conf["tp"] + conf["fn"])
    specificity = conf["tn"] / (conf["tn"] + conf["fp"])
    precision = conf["tp"] / (conf["tp"] + conf["fp"])
    f1_score = (2 * precision * recall) / (precision + recall)
    perf_metrics = {"acc": acc,
                    "specificity": specificity, "f1_score": f1_score}

    return perf_metrics


result_test = knn(1, 37, 296, 0, 37, train, "v1")
print(result_test)
print(len(result_test))
# lt = 37
# rt = 296
# lv = 0
# rv = 37
