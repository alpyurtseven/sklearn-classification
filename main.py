import  matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class YapayZeka():

    def __init__(self):

        self.data = pd.read_csv("Kod\data.csv")
        self.prepareData()
        f1=pd.read_csv("Kod\\f1.csv")
        #self.plot4()
        #self.logisticRegression()
        #self.knnClassification()
        #self.naiveBayesClassification()
        #self.decisionTreesClassification()
        #self.neuralNetworkClassification()
        sns.catplot(x="class", y="f1", 
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=f1)
        plt.show()


    def plot1(self):
        plot = sns.countplot(x = "target", data = self.data, palette = "Set3")
        plot.set(xlabel = "Hastalık Durumu", ylabel = "Sayı", title = "Hastalık Durumu")
        plt.show()

    def plot2(self):

        plot = sns.countplot(x = "sex", hue = "target" , data = self.data, palette = "Set3")
        plot.set(xlabel = "Cinsiyet", ylabel = "Sayı", title = "Cinsiyete göre hastalık durumu")
        plot.legend("","")

        plt.show()

    def plot3(self):

        plot = sns.countplot(x = "age", data = self.data, palette = "Set3")
        plot.set(xlabel = "Yaş", ylabel = "Sayı", title = "Yaş Dağılımı")

        plt.show()

    def plot4(self):

        hasta = self.data.loc[self.data.target == 1]
        plot = sns.countplot(x = "age",hue = "target", data = hasta, palette = "Set3")
        plot.set(xlabel = "Yaş", ylabel = "Sayı", title = "Hasta olanların yaş dağılımı")
        plot.legend("")

        plt.show()

    def prepareData(self):

        self.X = self.data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "exang", "oldpeak", "slope", "ca", "thal"]]
        self.Y = self.data["target"]
        standardScaler = StandardScaler()
        self.X=standardScaler.fit_transform(self.X)
        self.X=standardScaler.transform(self.X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=1)

    def logisticRegression(self):

        model = LogisticRegression(max_iter=200)
        model.fit(self.X_train,self.Y_train)
        prediction = model.predict(self.X_test)

        self.plotConfusionMatrix(prediction, model)
        print(f1_score(self.Y_test, prediction))

    def knnClassification(self):

        model = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
        model.fit(self.X_train,self.Y_train)
        prediction = model.predict(self.X_test)

        self.plotConfusionMatrix(prediction, model)
        print(f1_score(self.Y_test, prediction))

    def naiveBayesClassification(self):

        model = GaussianNB()
        model.fit(self.X_train, self.Y_train)
        prediction = model.predict(self.X_test)
    
        self.plotConfusionMatrix(prediction, model)
        print(f1_score(self.Y_test, prediction))

    def decisionTreesClassification(self):

        model = DecisionTreeClassifier() 
        model.fit(self.X_train, self.Y_train)
        prediction = model.predict(self.X_test)

        self.plotConfusionMatrix(prediction, model)
        print(f1_score(self.Y_test, prediction))

    def neuralNetworkClassification(self):

        model = MLPClassifier(random_state=1,max_iter=500)
        model.fit(self.X_train,self.Y_train)
        prediction = model.predict(self.X_test)
       
        self.plotConfusionMatrix(prediction, model)
        print(f1_score(self.Y_test, prediction))

    

    def plotConfusionMatrix(self,pred,model):

        confusionMatrix = confusion_matrix(self.Y_test, pred)
        plot_confusion_matrix(model,self.X_test, self.Y_test, display_labels=["(0) Sağlıklı","(1) Hasta"])


        TP = confusionMatrix[0][0]
        FP = confusionMatrix[1][0]
        TN = confusionMatrix[1][1]
        FN = confusionMatrix[0][1]

        print("Sensivity")   #TP/TP+FN
        print(TP/(TP+FN))

        print("Accuracy")    #(TP+TN)/(TP+TN+FP+FN)
        print((TP+TN)/(TP+FP+TN+FN))

        print("Specificity") #TN/TN+FP
        print(TN/(TN+FP))

        print("Recall")      #TP/TP+FN =Sensivity
        print(TP/(TP+FN))

        print("Precision")   #TP/(TP+FP)
        print(TP/(TP+FP))

        plt.show()
 
YapayZeka()
    


