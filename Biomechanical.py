import mlflow
import mlflow.sklearn

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# plotly
from chart_studio import plotly
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Data/column_2C_weka.csv")
print(df.head())

# Display the content of data
print(df.info())

print(df.describe())


#f, ax = plt.subplots(figsize=(10,10))
#sns.heatmap(df.corr(), annot=True, linewidth=".5", cmap="RdPu", fmt=".2f", ax = ax)
#plt.title("Correlation Map",fontsize=20)
#plt.show()

#sorts all correlations with ascending sort.
print(df.corr().unstack().sort_values().drop_duplicates())


### Preprocessing
df["class"] = [0 if each == "Abnormal" else 1 for each in df["class"]]

y = df["class"].values
x_data = df.drop(["class"], axis=1)

### Data normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


with mlflow.start_run():
    num_vicini=22
    knn_model = KNeighborsClassifier(n_neighbors = num_vicini)
    knn_model.fit(x_train,y_train)
    y_pred = knn_model.predict(x_test)
    y_true = y_test
    report = classification_report(y_true, y_pred, output_dict=True)


    macro_precision =  report['macro avg']['precision'] 
    macro_recall = report['macro avg']['recall']    
    macro_f1 = report['macro avg']['f1-score']
    acc = report["accuracy"]

    mlflow.sklearn.log_model(knn_model, "model")

    mlflow.log_param("n_neighbors", num_vicini)

    mlflow.log_metric("macro_precision", macro_precision)
    mlflow.log_metric("macro_recall", macro_recall)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("acc", acc)


    # Confusion Matrix
    cm_lr = confusion_matrix(y_true,y_pred)
    f, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm_lr, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
    plt.xlabel = ('y_pred')
    plt.ylabel = ('y_true')
    plt.show()
    mlflow.log_figure(f, "figure.png")




'''### Checking Accuracy

# Model complexity
neighboors = np.arange(1,30)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neighboors):
    # k from 1 to 30(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit with knn
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))           # train accuracy
    test_accuracy.append(knn.score(x_test, y_test))              # test accuracy
# import graph objects as "go"
import plotly.graph_objs as go
# Creating trace1
trace1 = go.Scatter(
                    x = neighboors,
                    y = train_accuracy,
                    mode = "lines",
                    name = "train_accuracy",
                    marker = dict(color = 'rgba(160, 112, 2, 0.8)'),
                    text= "train_accuracy")
# Creating trace2
trace2 = go.Scatter(
                    x = neighboors,
                    y = test_accuracy,
                    mode = "lines+markers",
                    name = "test_accuracy",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "test_accuracy")
data = [trace1, trace2]
layout = dict(title = 'K Value vs Accuracy',
              xaxis= dict(title= 'Number of Neighboors',ticklen= 10,zeroline= True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
knn_train_accuracy = np.max(train_accuracy)
knn_test_accuracy = np.max(test_accuracy)
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))'''



















