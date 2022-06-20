import pandas as pd
import numpy as np
import math
import random
import time
from icecream import ic


def knn(k, lv, rv, train):
    result_validation = []
    result_training = []
    for i in range(len(train.index)):
        result_v = []
        result_t = []
        for j in range(len(train.index)):
            if i >= lv and i < rv:
                if j >= lv and j < rv:
                    result_v.append(np.inf)
                else:
                    sum = math.sqrt(((train["x1"][i] - train["x1"][j]) ** 2) + ((train["x2"]
                                                                                 [i] - train["x2"][j]) ** 2) + ((train["x3"][i] - train["x3"][j]) ** 2))
                    # sum = abs(train["x1"][i] - train["x1"][j]) + abs(train["x2"][i] -
                    #                                                  train["x2"][j]) + abs(train["x3"][i] - train["x3"][j])
                    result_v.append(sum)
            else:
                if (j >= lv and j < rv) or i == j:
                    result_t.append(np.inf)
                else:
                    sum1 = math.sqrt(((train["x1"][i] - train["x1"][j]) ** 2) + ((train["x2"]
                                                                                 [i] - train["x2"][j]) ** 2) + ((train["x3"][i] - train["x3"][j]) ** 2))
                    # sum1 = abs(train["x1"][i] - train["x1"][j]) + abs(train["x2"][i] -
                    #                                                   train["x2"][j]) + abs(train["x3"][i] - train["x3"][j])
                    result_t.append(sum1)

        if i >= lv and i < rv:
            nearest_v = []
            for _ in range(k):
                idx_v = result_v.index(min(result_v))
                select = train.loc[idx_v]
                nearest_v.append(select["y"])
                result_v[idx_v] = np.inf
            if nearest_v.count(1) > nearest_v.count(0):
                result_validation.append(1)
            elif nearest_v.count(0) > nearest_v.count(1):
                result_validation.append(0)
            else:
                result_validation.append(random.choice([0, 1]))
        else:
            for _ in range(k):
                nearest_t = []
                idx_t = result_t.index(min(result_t))
                select = train.loc[idx_t]
                nearest_t.append(select["y"])
                result_t[idx_t] = np.inf

            if nearest_t.count(1) > nearest_t.count(0):
                result_training.append(1)
            elif nearest_t.count(0) > nearest_t.count(1):
                result_training.append(0)
            else:
                result_training.append(random.choice([0, 1]))
    # print("validation :", len(result_validation))
    # print("test :", len(result_training))

    return result_training, result_validation


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


def validation(k, train):
    result_avg_v = []
    result_f1_v = []
    result_avg_t = []
    result_f1_t = []
    for i in range(1, 9, 1):
        lv = (len(train.index)*(i-1))//8
        rv = (len(train.index)*(i))//8
        result_yt, result_yv = knn(k, lv, rv, train)
        tp_v = 0
        tn_v = 0
        fp_v = 0
        fn_v = 0
        for j in range(len(result_yv)):
            if result_yv[j] == 1:
                if train["y"][j+lv] == 1:
                    tp_v += 1
                else:
                    fp_v += 1
            elif result_yv[j] == 0:
                if train["y"][j+lv] == 0:
                    tn_v += 1
                else:
                    fn_v += 1
        confusion_matrix_v = {"tp": tp_v, "tn": tn_v, "fp": fp_v, "fn": fn_v}
        perf_v = performance(confusion_matrix_v)
        result_avg_v.append(perf_v["acc"])
        result_f1_v.append(perf_v["f1_score"])
        tp_t = 0
        tn_t = 0
        fp_t = 0
        fn_t = 0
        if i == 1:
            for l in range(len(result_yt)):
                if result_yt[l] == 1:
                    if train["y"][l+rv] == 1:
                        tp_t += 1
                    else:
                        fp_t += 1
                elif result_yt[l] == 0:
                    if train["y"][l+rv] == 0:
                        tn_t += 1
                    else:
                        fn_t += 1
        elif i == 8:
            for l in range(len(result_yt)):
                if result_yt[l] == 1:
                    if train["y"][l] == 1:
                        tp_t += 1
                    else:
                        fp_t += 1
                elif result_yt[l] == 0:
                    if train["y"][l] == 0:
                        tn_t += 1
                    else:
                        fn_t += 1
        else:
            for l in range(lv):
                if result_yt[l] == 1:
                    if train["y"][l] == 1:
                        tp_t += 1
                    else:
                        fp_t += 1
                elif result_yt[l] == 0:
                    if train["y"][l] == 0:
                        tn_t += 1
                    else:
                        fn_t += 1
            for m in range(lv, len(result_yt), 1):
                if result_yt[m] == 1:
                    if train["y"][m+(len(train.index)//8)] == 1:
                        tp_t += 1
                    else:
                        fp_t += 1
                elif result_yt[m] == 0:
                    if train["y"][m+(len(train.index)//8)] == 0:
                        tn_t += 1
                    else:
                        fn_t += 1
        confusion_matrix_t = {"tp": tp_t, "tn": tn_t, "fp": fp_t, "fn": fn_t}
        perf_t = performance(confusion_matrix_t)
        result_avg_t.append(perf_t["acc"])
        result_f1_t.append(perf_t["f1_score"])

    result_v = {"acc": np.mean(np.array(result_avg_v)),
                "f1_score": np.mean(np.array(result_f1_v))}

    result_t = {"acc": np.mean(np.array(result_avg_t)),
                "f1_score": np.mean(np.array(result_f1_t))}

    return result_t, result_v


if __name__ == "__main__":
    st = time.time()
    # Use your own path file
    train = pd.read_excel(
        r'traintest.xlsx', sheet_name="train")

    test = pd.read_excel(
        r'traintest.xlsx', sheet_name="test")

    x = train.iloc[:, 1:4]
    y = test.iloc[:, 1:4]
    train.iloc[:, 1:4] = (x-x.min()) / (x.max() - x.min())
    test.iloc[:, 1:4] = (y-x.min()) / (x.max() - x.min())

    # Shuffle the data so hopefully the class is evenly distributed
    train = train.sample(frac=1)

    min_k = 0
    difference = np.inf
    for i in range(1, 11, 1):
        result_t, result_v = validation(i, train)
        print("K =", i)
        print("Training")
        ic("Accuracy :", result_t["acc"],
           "F1-Score :", result_t["f1_score"])
        print("Validation")
        ic("Accuracy :", result_v["acc"],
           "F1-Score :", result_v["f1_score"])
        if (abs(result_t["acc"] - result_v["acc"]) < difference) and result_t["acc"] > result_v["acc"]:
            min_k = i
            difference = abs(result_t["acc"] - result_v["acc"])
   # result_t["acc"] >= 0.65 and result_v["acc"] >= 0.7

    print()
    et = time.time()
    elapsed = et - st
    print("Elapsed time :", elapsed)

    result_y = []
    for i in range(len(test.index)):
        result = []
        for j in range(len(train.index)):
            sum = math.sqrt(((test["x1"][i] - train["x1"][j]) ** 2) + ((test["x2"]
                                                                        [i] - train["x2"][j]) ** 2) + ((test["x3"][i] - train["x3"][j]) ** 2))
            # sum = abs(test["x1"][i] - train["x1"][j]) + abs(test["x2"][i] -
            #                                                 train["x2"][j]) + abs(test["x3"][i] - train["x3"][j])
            result.append(sum)
        nearest = []
        for _ in range(min_k):
            idx = result.index(min(result))
            select = train["y"][idx]
            nearest.append(select)
            result[idx] = np.inf
        if nearest.count(1) > nearest.count(0):
            result_y.append(1)
        elif nearest.count(0) > nearest.count(1):
            result_y.append(0)
        else:
            result_y.append(random.choice([0, 1]))

    print("k value :", min_k)
    ic(result_y)
