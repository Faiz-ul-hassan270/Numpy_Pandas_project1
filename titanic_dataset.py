# -------------------- LIBRARIES IMPORT --------------------
import pandas as pd   # pandas library is imported for data handling and analysis
import numpy as np    # numpy library is imported for numerical operations

# -------------------- READ DATASET --------------------
df = pd.read_csv("tested.csv", encoding="utf-8") 
# csv file ko read karne ke liye aur usse df (DataFrame) me store kiya

# -------------------- DATA PREVIEW --------------------
print(" TOP 10 ROWS ".center(80,"="))
print(f"\n{df.head(10)}\n")   # top 10 rows preview

print(" LAST 10 ROWS ".center(80,"="))
print(f"\n{df.tail(10)}\n")   # last 10 rows preview

# -------------------- DATASET STRUCTURE --------------------
print(" DATAFRAME SHAPE ".center(80,"="))
print(f"\n{df.shape}\n")   # total rows × columns

print(" COLUMNS NAMES ".center(80,"="))
print(f"\n{df.columns}\n")   # all column names

print(" DATAFRAME INFO ".center(80,"="))
print(f"\n{df.info()}\n")   # summary of dataset (columns, types, nulls)

print(" DATAFRAME DTYPES ".center(80,"="))
print(f"\n{df.dtypes}\n")   # data types of each column

# -------------------- MISSING VALUES HANDLING --------------------
print(" MISSING VALUES ".center(80,"="))
print(f"\n{df.isnull().sum()}\n")   # count missing values in each column

age_median = df['Age'].median() 
print(" FILLING AGE NAN WITH MEDIAN ".center(80,"="))
print(f"\n{df.fillna({'Age': age_median}, inplace=True)}\n") 
# missing values in 'Age' are filled with median

Cabin_mode = df['Cabin'].mode()[0]
print(" FILLING CABIN NAN WITH MODE ".center(80,"="))
print(f"\n{df['Cabin'].fillna(Cabin_mode, inplace=True)}\n") 
# missing values in 'Cabin' are filled with most frequent value (mode)

# -------------------- DUPLICATES --------------------
print(" DUPLICATES COUNT ".center(80,"="))
print(f"\nDuplicates are : {df.duplicated().sum()}\n")   # duplicate rows count

# -------------------- VALUE COUNTS --------------------
print(" SEX VALUE COUNTS ".center(80,"="))
print(f"\n{df['Sex'].value_counts()}\n")   # frequency of male/female

print(" SURVIVAL VALUE COUNTS ".center(80,"="))
print(f"\n{df['Survived'].value_counts()}\n")   # count of survived/not survived

# -------------------- STATISTICS --------------------
print(" AGE STATISTICS ".center(80,"="))
print(f"The minimum age is : {df['Age'].min()}")      # min age
print(f"The maximum age is : {df['Age'].max()}")      # max age
print(f"The average age is : {df['Age'].mean()}")     # mean age
print(f"The median of age is : {df['Age'].median()}") # median age

print(" FARE STATISTICS ".center(80,"="))
print(f"The minimum fare is : {df['Fare'].min()}")      # min fare
print(f"The maximum fare is : {df['Fare'].max()}")      # max fare
print(f"The average fare is : {df['Fare'].mean()}")     # mean fare
print(f"The median of fare is : {df['Fare'].median()}") # median fare

# -------------------- CLASS COUNTS --------------------
print(" PASSENGERS CLASS COUNTS ".center(80,"="))
print(f"\n{df['Pclass'].value_counts().sort_index()}\n")   # count per passenger class

# -------------------- FILTERING DATA --------------------
print(" CHILDREN PASSENGERS ".center(80,"="))
print(f"\n{df[df['Age'] < 18]}\n")   # passengers below 18

print(" SURVIVED FEMALE PASSENGERS ".center(80,"="))
print(f"\n{df[(df['Sex']=='female') & (df['Survived']==1)]}\n")   # survived females

print(" SURVIVED MALE PASSENGERS ".center(80,"="))
print(f"\n{df[(df['Sex']=='male') & (df['Survived']==1)]}\n")   # survived males

# -------------------- SORTING --------------------
print(" AGE WISE SORT ASCENDING ".center(80,"="))
print(f"\n{df.sort_values(by='Age', ascending=True)}\n")   # age ascending

print(" FARE WISE SORT DESCENDING ".center(80,"="))
print(f"\n{df.sort_values(by='Fare', ascending=False)}\n") # fare descending

print(" TOP 5 OLDEST PASSENGERS ".center(80,"="))
oldest_5 = df.sort_values(by='Age', ascending=False).head(5) 
print(f"\n{oldest_5}\n")   # oldest 5 passengers

print(" TOP 5 RICHEST PASSENGERS ".center(80,"="))
richest_5 = df.sort_values(by="Fare", ascending=False).head(5) 
print(f"\n{richest_5}\n")   # richest 5 passengers

# -------------------- NUMPY STATS --------------------
print(" AGE NUMPY STATS ".center(80,"="))
age_array = df['Age'].to_numpy()
print(f"The average age is: {age_array.mean()}")    # mean using numpy
median_age = np.median(age_array)                   # median using numpy
print(f"The median age is: {median_age}")
std_age = np.std(age_array)                         # standard deviation
print(f"The standard deviation of age is: {std_age}")

print(" FARE NUMPY STATS ".center(80,"="))
fare_array = df["Fare"].dropna().to_numpy()
fare_mean = np.mean(fare_array)
print(f"The mean of fare is: {fare_mean:.2f}")
fare_median = np.median(fare_array)
print(f"The median of fare is: {fare_median}")
fare_std = np.median(fare_array) # (galti hai – yahan std nikalna tha median likha hai)
print(f"The std of fare is: {fare_std}")

# -------------------- ADVANCED FILTERS --------------------
print(" PASSENGERS AGE > 30 ".center(80,"="))
filter_age = df[df['Age'] > 30] 
print(f"\n{filter_age}\n")   # passengers older than 30

print(" AVG FARE OF SURVIVED PASSENGERS ".center(80,"="))
avg_fare = df[df["Survived"] == 1]["Fare"].mean() 
print(f"The avg fare is: {avg_fare}")   # average fare for survivors

# -------------------- AGE GROUPING + SURVIVAL --------------------
print(" SURVIVAL COUNTS BASED ON AGE GROUPS ".center(80,"="))
def age_group(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Old'
df['Age_group'] = df['Age'].apply(age_group) 
survival_counts = df.groupby(['Age_group','Survived']).size().unstack(fill_value=0)
survival_counts.columns = ['Not_Survived','Survived']
print(f"\n{survival_counts}\n")

# -------------------- CLASS WISE SURVIVAL RATE --------------------
print(" CLASS-WISE SURVIVAL RATE ".center(80,"="))
first_class = df[df['Pclass']==1]
first_class_survival_rate = (first_class['Survived'].sum() / len(first_class))*100
print(f"The first_class_survival_rate is: {first_class_survival_rate}")

second_class = df[df['Pclass']==2]
second_class_survival_rate = (second_class['Survived'].sum() / len(second_class))*100
print(f"The second_class_survival_rate is: {second_class_survival_rate}")

third_class = df[df['Pclass']==3]
third_class_survival_rate = (third_class['Survived'].sum() / len(third_class))*100
print(f"The third_class_survival_rate is: {third_class_survival_rate}")

# -------------------- CLASS + GENDER SURVIVAL --------------------
print(" GENDER + CLASS SURVIVAL RATE ".center(80,"="))
groups = df.groupby(['Pclass','Sex'])
for (key,value), group in groups:
    rate = (group['Survived'].sum() / len(group)) * 100
    print(f"Class {key}, Gender {value} → Survival Rate: {rate}%")

# -------------------- CLASS + EMBARKED --------------------
print(" CLASS + EMBARKED AVG FARE ".center(80,"="))
p_groups = df.groupby(['Pclass','Embarked'])
for (key,value), group in p_groups:
    average = group['Fare'].mean()
    print(f"Class {key}, Embarked {value} → Avg Fare: {average}")

# -------------------- EXTRA FEATURES --------------------
print(" CABIN MISSING VALUES ".center(80,"="))
print(f"Missing values in cabin column are: {df['Cabin'].isnull().sum()}")

print(" FAMILY SIZE & ALONE PASSENGERS ".center(80,"="))
df['FamilySize'] = df['SibSp'] + df['Parch']   # family size = siblings/spouses + parents/children
alone_passengers = df[df['FamilySize']==0]
print(f"Alone passengers are : {len(alone_passengers)}")

print(" UNIQUE TICKETS ".center(80,"="))
print(f"Unique values in tickets are : {len(df['Ticket'].unique())}")

print(" AGE INTERPOLATION ".center(80,"="))
print(f"The missing values in age before interpolation: {df['Age'].isnull().sum()}")
age_interpolate = df['Age'].interpolate(method='linear')
print(f"The missing values in age after interpolation: {df['Age'].isnull().sum()}")

print(" FARE OUTLIER CHECK ".center(80,"="))
print(f"\n{df['Fare'].describe()}\n")   # describe shows stats → used for outlier detection
