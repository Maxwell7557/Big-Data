import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def makeCharts(table) :
    firstPivot = table.pivot_table(values='PassengerId',index='Sex',columns='Survived',aggfunc='count')
    firstPivot.plot(title='Sex -> Survived',kind='bar',stacked=True)
    # plt.show()

    secondPivot = table.pivot_table(values='PassengerId',index='Age',columns='Survived',aggfunc='count')
    secondPivot.plot(title='Age -> Survived',kind='area',stacked=True)
    # plt.show()

    thirdPivot = table.pivot_table(values='PassengerId',index='Parch',columns='Survived',aggfunc='count')
    thirdPivot.plot(title='Parch -> Survived',kind='bar',logy=True,stacked=False)
    # plt.show()

    fourthPivot = table.pivot_table(values='PassengerId',index='SibSp',columns='Survived',aggfunc='count')
    fourthPivot.plot(title='SibSp -> Survived',kind='bar',logy=True,stacked=False)
    # plt.show()

    table['Embarked'].replace('C','France',inplace=True)
    table['Embarked'].replace('S','England',inplace=True)
    table['Embarked'].replace('Q','Ireland',inplace=True)
    fithPivot = table.pivot_table(values='PassengerId',index='Embarked',columns='Survived',aggfunc='count')
    fithPivot.plot(title='Embarked -> Survived',kind='bar',stacked=False)
    # plt.show()

    table['Fare'] = table['Fare'].astype('int')
    sixPivot = table.pivot_table(values='PassengerId',index='Fare',columns='Survived',aggfunc='count')
    sixPivot.plot(title='Fare -> Survived',kind='area',logx=True,stacked=False)
    plt.show()

if __name__ == "__main__" :
    table = pd.read_csv('/home/maxwell/Big Data/titanic/train.csv')

    makeCharts(table)
