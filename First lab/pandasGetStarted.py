import pandas as pd

if __name__ == "__main__" :
    table = pd.read_csv('/home/maxwell/Big Data/titanic/train.csv',index_col='PassengerId')

    print('*'*10)
    genderTable = table['Sex'].value_counts()
    print(f'Male:   {genderTable[0]}',f'Female: {genderTable[1]}',sep='\n',end='\n**********\n')#first part

    survived = table['Survived'].value_counts()
    percentOfSurvived = (survived[1] * 100) / (survived[0] + survived[1])
    print(f'Percent of survived:    {percentOfSurvived}%',end='\n**********\n') #second part

    classTable = table['Pclass'].value_counts()
    percentOfClasses = (classTable[1] * 100) / (classTable[1] + classTable[2] + classTable[3])
    print(f'Percent of first class:    {percentOfClasses}%',end='\n**********\n') #third part

    print(f'Mean value: {table["Age"].mean()}',end='\n**********\n')
    print(f'Median value: {table["Age"].median()}',end='\n**********\n')

    table.replace('.*\s(Mlle.\s|Miss.\s\(?|(Mrs.|Mme.|Ms.|Countess.|Dr.)((\w|\s)*\(|\s))', '', regex=True,inplace=True)
    table.replace('(\s.*|\)?)', '', regex=True,inplace=True)
    femaleTable = table[table.Sex == 'female']['Name'].value_counts()
    print(f'List of female names (from most popular to less popular):',femaleTable,sep='\n')
