# Numpy_Pandas_project1
# Titanic Data Analysis Project

## 🚀 Project Overview
This project focuses on **data analysis using Python** with **NumPy** and **Pandas** on the famous **Titanic dataset** (`tested.csv`). The main goal was to practice **data cleaning, exploration, and feature extraction**, while generating meaningful insights from real-world data.

---

## 🔍 Key Steps Performed

1. **Dataset Loading & Exploration**  
   - Checked the dataset shape, column names, and data types.  
   - Examined missing values and duplicates.

2. **Handling Missing Values**  
   - Filled missing values using **median**, **mode**, and **linear interpolation** methods.

3. **Duplicate Detection & Removal**  
   - Checked for duplicate rows and handled them.

4. **Statistical Analysis**  
   - Calculated **minimum, maximum, mean, median, and standard deviation** for `Age` and `Fare`.

5. **Data Filtering & Subsets**  
   - Filtered children passengers (`Age < 18`).  
   - Extracted **survived female** and **survived male** passengers.  

6. **Survival Analysis**  
   - Calculated survival rates by **gender**, **class**, and **age groups** (Child, Adult, Old).  

7. **Feature Engineering**  
   - Created **Age Groups** (`Child`, `Adult`, `Old`) and **Family Size** columns.  

8. **Outlier Detection**  
   - Identified outliers in `Fare` using **IQR method**.

---

## 📊 Insights from Analysis

- Female passengers had a higher survival rate compared to males.  
- First-class passengers had the highest survival rate among all classes.  
- Children had better chances of survival compared to adults and older passengers.  
- Many passengers traveled alone (FamilySize = 0).  
- Fare distribution showed strong outliers among a few richest passengers.

---

## 🧰 Tools & Libraries Used

- **Python**  
- **NumPy**  
- **Pandas**

---

## 📚 Learning Outcomes

Through this project, I improved my skills in:  
- Data cleaning and handling missing values  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Filtering and analyzing data subsets  

---

## 🔗 GitHub Repository

Full code is available here: [GitHub Link]  

---

## 💡 Discussion

How do you usually handle missing values in your datasets — **Median/Mode**, **Interpolation**, or **Drop strategy**? Share your approach in the comments!

