"""Regenerate match_predict.joblib with the currently installed scikit-learn,
since the committed joblib file was pickled with an old sklearn version whose
internal tree format is incompatible with newer releases."""

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

users = pd.read_csv("users.csv")
swipes = pd.read_csv("swipes.csv")

users_new = users[["gender", "interests", "is_verified", "status", "is_subscribed", "createdAt", "updatedAt"]]
users_new = users_new.dropna()

users_new["is_verified"] = users_new["is_verified"].apply(lambda x: 0 if x == "False" else 1)
users_new["status"] = users_new["status"].apply(lambda x: 0 if x == "False" else 1)
users_new["is_subscribed"] = users_new["is_subscribed"].apply(lambda x: 0 if x == "False" else 1)

swipes_new = swipes[["is_unmatch", "p1", "p2", "first_type", "second_type"]]
swipes_new["is_unmatch"] = swipes_new["is_unmatch"].apply(lambda x: 1 if x == False else 0)
swipes_new["first_type"] = swipes_new["first_type"].apply(lambda x: 0 if x == "DISLIKE" else 1)
swipes_new["second_type"] = swipes_new["second_type"].apply(lambda x: 0 if x == "DISLIKE" else 1)
swipes_new = swipes_new.dropna()

l = []
y = []
for i in swipes_new.values:
    l.append([i[1], i[2]])
    y.append(i[0])

data = list(users_new["interests"])
list_of_lists = [json.loads(s) for s in data]
ints = list(itertools.chain(*list_of_lists))


def one_hot(lis):
    return [1 if value in lis else 0 for value in ints]


fm = pd.DataFrame(
    [],
    columns=["Gender1", "interests1", "is_verified1", "status1", "is_subscribed1",
             "Gender2", "interests2", "is_verified2", "status2", "is_subscribed2", "is_unmatch"],
)

for i in range(len(l)):
    try:
        lh = l[i][0]
        hh = l[i][1]
        row_lst = list(users_new.loc[lh][:5].values) + list(users_new.loc[hh][:5].values) + [y[i]]

        row_lst[0] = 1 if row_lst[0] == "F" else 0
        row_lst[5] = 1 if row_lst[5] == "F" else 0

        fm.loc[len(fm)] = row_lst
    except Exception:
        continue

data = list(fm["interests1"])
list_of_lists = [json.loads(s) for s in data]
fm["interests1"] = [one_hot(i) for i in list_of_lists]

data = list(fm["interests2"])
list_of_lists = [json.loads(s) for s in data]
fm["interests2"] = [one_hot(i) for i in list_of_lists]

expanded_df = fm["interests1"].apply(pd.Series)
fm1 = pd.concat([fm, expanded_df], axis=1).drop("interests1", axis=1)

expanded_df = fm1["interests2"].apply(pd.Series)
fm2 = pd.concat([fm1, expanded_df], axis=1).drop("interests2", axis=1)

X = fm2.drop("is_unmatch", axis=1).to_numpy()
y = fm2["is_unmatch"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

joblib.dump(clf, "match_predict.joblib")
print("Wrote match_predict.joblib")
