import pandas as pd

train = pd.read_excel(
    r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="train")

test = pd.read_excel(
    r'C:\Users\rifqi\OneDrive\Documents\Folder Tugas Iqi\Semester 4\Pengantar Kecerdasan Buatan\Learning Programming Assignment\traintest.xlsx', sheet_name="test")


test_id = test["id"]
train_id = train["id"]

test_x1 = test["x1"]
test_x2 = test["x2"]
test_x3 = test["x3"]

train_x1 = train["x1"]
train_x2 = train["x2"]
train_x3 = train["x3"]
train_y = train["y"]

print(len(test))

result_y = []
for i in range(len(test_id)):
    result = []
    for j in range(len(train_id)):
        sum = abs(test_x1[i] - train_x1[j]) + abs(test_x2[i] -
                                                  train_x2[j]) + abs(test_x3[i] - train_x3[j])
        result.append(sum)
    res = result.index(min(result))
    result_y.append(train["y"][res+1])

print(result_y)
