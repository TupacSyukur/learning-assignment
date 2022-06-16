import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time

# result_y = []
# for i in range(len(test_id)):
#     result = []
#     for j in range(len(train_id)):
#         sum = abs(test_x1[i] - train_x1[j]) + abs(test_x2[i] -
#                                                   train_x2[j]) + abs(test_x3[i] - train_x3[j])
#         result.append(sum)
#     res = result.index(min(result))
#     result_y.append(train["y"][res+1])


# Function knn should be reviewed again!!!
def knn(k, lv, rv, train):
    result_y = []
    for i in range(lv, rv, 1):
        result = []
        for j in range(296):
            if j >= lv and j < rv:  # for i in range(number of data)
                result.append(np.inf)
            else:
                sum = math.sqrt(((train["x1"][i] - train["x1"][j]) ** 2) + ((train["x2"]
                                                                             [i] - train["x2"][j]) ** 2) + ((train["x3"][i] - train["x3"][j]) ** 2))
                result.append(sum)
        nearest = []
        for i in range(k):
            idx = result.index(min(result))
            select = train.loc[idx]
            nearest.append(select["y"])
            result[idx] = np.inf
        if nearest.count(1) > nearest.count(0):
            result_y.append(1)
        elif nearest.count(0) > nearest.count(1):
            result_y.append(0)
        else:
            result_y.append(random.choice([0, 1]))

    return result_y


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
    result_avg = []
    result_f1 = []
    for i in range(1, 9, 1):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        lv = (296*(i-1))//8
        rv = (296*(i))//8
        result_y = knn(k, lv, rv, train)
        for j in range(len(result_y)):
            if result_y[j] == 1:
                if train["y"][j+lv] == 1:
                    tp += 1
                else:
                    fp += 1
            elif result_y[j] == 0:
                if train["y"][j+lv] == 0:
                    tn += 1
                else:
                    fn += 1
        confusion_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        perf = performance(confusion_matrix)
        result_avg.append(perf["acc"])
        result_f1.append(perf["f1_score"])

    result = {"acc": np.mean(np.array(result_avg)),
              "f1_score": np.mean(np.array(result_f1))}

    return result


if __name__ == "__main__":
    st = time.time()
    # Use your own path file
    train = pd.read_excel(
        r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="train")

    test = pd.read_excel(
        r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="test")

    x = train.iloc[:, 1:4]
    train.iloc[:, 1:4] = (x-x.min()) / (x.max() - x.min())

    plots = []
    for i in range(1, 51, 1):
        result = validation(i, train)
        print("K =", i)
        print("Accuracy :", result["acc"])
        plots.append(result["acc"])
    print()
    et = time.time()
    elapsed = et - st
    print("Elapsed time :", elapsed)

    plt.plot(plots)
    plt.ylabel("Accuracy")
    plt.show()
