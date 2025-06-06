# importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# loading the dataset
df = pd.read_csv('D:\heart.csv')

# แสดงหัวตารางข้อมูลแรก
# แสดงหัวตารางข้อมูลท้าย
# แสดงรูปร่างของข้อมูล (จำนวนแถว, จำนวนคอลัมน์)
df.head()
print('Head\n', df.head())
print('Tail\n', df.tail())
print('Shape\n', df.shape)


# Generating descriptive statistics. แสดงค่าสถิติทางคณิตศาสตร์ของข้อมูล
print('describe\n', df.describe().T)

# นับจำนวนผู้ป่วยที่มีและไม่มีโรคหัวใจ
df['target'] = df['target'].replace({0: "Doesn't have heart disease", 1: "Has Heart Disease"})
print('values\n', df.target.value_counts())

# สร้างกราฟ Pie chart เพื่อแสดงสัดส่วนของผู้ป่วยที่มีและไม่มีโรคหัวใจ                1
labels = "Has heart disease", "Doesn't have heart disease"
explode = (0, 0)
colors = ['lightcoral', 'palegreen']

fig1, ax1 = plt.subplots()
ax1.pie(df.target.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')
plt.show()

# fining the gender : นับจำนวนผู้ป่วยเพศชายและเพศหญิง
print('Gender\n', df.sex.value_counts())

# visualizing in Pie chart : สร้างกราฟ Pie chart เพื่อแสดงสัดส่วนของเพศชายและเพศหญิง        2
labels = 'Male', 'Female'
explode = (0, 0)
colors = ['steelblue', 'lightpink']

fig1, ax1 = plt.subplots()
ax1.pie(df.sex.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')
plt.show()

# สร้างตาราง Crosstab แสดงความสัมพันธ์ระหว่างเป้าหมาย (โรคหัวใจ) และเพศ
print('Target and Gender\n', pd.crosstab(df.target, df.sex))

# Frequency for Gender : สร้างกราฟ Countplot เพื่อแสดงความถี่ของโรคหัวใจแยกตามเพศ           3
palette = ['lightpink', 'steelblue']
fig = sns.countplot(x='target', data=df, hue='sex', palette=palette)
fig.set_xticklabels(labels=["Doesn't have heart disease", 'Has heart disease'], rotation=0)
plt.legend(['Female', 'Male'])
plt.title("Heart Disease Frequency for Sex")
plt.show()


# สร้าง scatter plot หรือ swarm plot แสดงความสัมพันธ์ระหว่างอายุกับผู้ป่วยที่มีและไม่มีโรคหัวใจ        4
df['target'] = df['target'].replace({0: "Doesn't have heart disease"})
df['target'] = df['target'].replace({1: "Has heart disease"})
plt.figure(figsize=(12, 6))
sns.swarmplot(x='target', y='age', data=df, palette=['lightcoral', 'green'])

# เพิ่มชื่อแกน x และ y และตั้งชื่อกราฟ
plt.ylabel('Age')
plt.title('Age Convert Heart Disease Relationship')
plt.show()


# visualizing in Pie chart        แสดงระดับน้ำตาลในเลือด                      5
labels = 'fbs<120 mg/dl', 'fbs>120 mg/dl'
explode = (0, 0)
colors = ['steelblue', 'lightpink']

fig1, ax1 = plt.subplots()
ax1.pie(df.fbs.value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')
plt.show()

# visualizing fbs     การวิเคราะห์ความสัมพันธ์ระหว่างระดับน้ำตาลในเลือดกับเพศ                    6
fig = pd.crosstab(df.sex, df.fbs).plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title("Fasting blood sugar w.r.t sex")
fig.set_xticklabels(labels=['fbs>120 mg/dl', 'fbs<120 mg/dl'], rotation=0)
plt.legend(['Female', 'Male'])
plt.show()

# ตรวจดูว่าข้อมูลใน  เป็นต่อเนื่องหรือไม่ต่อเนื่อง
data_type = df['age'].dtype
if pd.api.types.is_float_dtype(data_type) or pd.api.types.is_integer_dtype(data_type):
    print("คอลัมน์ 'age' เป็นข้อมูลต่อเนื่อง")
else:
    print("คอลัมน์ 'age' ไม่เป็นข้อมูลต่อเนื่อง")

# สร้างเมทริกซ์ Features และ Target
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# แบ่งชุดข้อมูลเป็น Train และ Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=31)

# หาจำนวน sample ใน Train set และ Test set
num_samples_train = x_train.shape[0]
num_samples_test = x_test.shape[0]

print("จำนวน sample ใน Train set:", num_samples_train)
print("จำนวน sample ใน Test set:", num_samples_test)

# สร้างแบบจำลอง Logistic Regression
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter=1000, random_state=4)
log_clf.fit(x_train, y_train)
log_score = log_clf.score(x_test, y_test)
print('Regression\n', log_score)

# สร้างแบบจำลอง KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
k_value = 5  # ตั้งค่า k ที่คุณต้องการ
knn_clf = KNeighborsClassifier(n_neighbors=k_value)
knn_clf.fit(x_train, y_train)
knn_score = knn_clf.score(x_test, y_test)
print('KNN\n', knn_score)

# สร้างแบบจำลอง Support Vector Classifier SVM Classification Report
from sklearn import svm
svc_clf = svm.SVC(random_state=7)
svc_clf.fit(x_train, y_train)
svc_score = svc_clf.score(x_test, y_test)
# print('SVC\n', svc_score)

# สร้างแบบจำลอง GradientBoostingClassifier GBC Classification Report
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_score = gbc.score(x_test, y_test)
print('GBC\n', gbc_score)

# สร้างแบบจำลอง Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
nb_score = nb_clf.score(x_test, y_test)
print('Gaussian Naive Bayes\n', nb_score)


# สร้างแบบจำลอง Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
dt_clf = DecisionTreeClassifier(random_state=7)
dt_clf.fit(x_train, y_train)
dt_score = dt_clf.score(x_test, y_test)
print('Decision Tree\n', dt_score)

anova = SelectKBest(f_classif, k=4)
x_train_anova = anova.fit_transform(x_train, y_train)
x_test_anova = anova.transform(x_test)
dt_clf_anova = DecisionTreeClassifier(random_state=43)
dt_clf_anova.fit(x_train_anova, y_train)
dt_score_anova = dt_clf_anova.score(x_test_anova, y_test)
print('Decision Tree (ANOVA)\n', dt_score_anova)

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
selected_features_desition_tree_anova = anova.get_support()
selected_feature_names = [names[i] for i, selected in enumerate(selected_features_desition_tree_anova) if selected]
print(f'Selected Features: {selected_feature_names}')

from sklearn.neural_network import MLPClassifier
# สร้างแบบจำลอง Multilayer Perceptron (MLP)
mlp_clf = MLPClassifier(random_state=7)
mlp_clf.fit(x_train, y_train)
mlp_score = mlp_clf.score(x_test, y_test)
print('Multilayer Perceptron\n', mlp_score)

# สร้างคุณลักษณะที่ถูกเลือกด้วย ANOVA
anova = SelectKBest(f_classif, k=4)
x_train_anova = anova.fit_transform(x_train, y_train)
x_test_anova = anova.transform(x_test)
# สร้างแบบจำลอง Multilayer Perceptron (MLP) ด้วยคุณลักษณะที่ถูกเลือกด้วย ANOVA
mlp_clf_anova = MLPClassifier(random_state=43)
mlp_clf_anova.fit(x_train_anova, y_train)
mlp_score_anova = mlp_clf_anova.score(x_test_anova, y_test)
print('Multilayer Perceptron (ANOVA)\n', mlp_score_anova)

# Making predictions on test set ทำนายผลลัพธ์บนชุดทดสอบแบบธรรมดา
y_preds = mlp_clf.predict(x_test)

# Making predictions on test set ANOVA ทำนายผลลัพธ์บนชุดทดสอบ
# y_preds = dt_clf_anova.predict(x_test_anova)

# confusion matrix
from sklearn.metrics import confusion_matrix
conF = confusion_matrix(y_test, y_preds)
print('Confusion matrix\n', conF)
# Plot the confusion matrix as a heatmap                     7
plt.figure(figsize=(8, 6))

print('Classification Report')
sns.heatmap(conF, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# classification_report : สร้างรายงานการจำแนกประเภท
from sklearn.metrics import classification_report
print(classification_report(y_test, y_preds))
score = [{'Model':'Logistic Regression', 'Score': log_score},
         {'Model':'KNN', 'Score': knn_score},
         {'Model':'Gradient Boosting Classifier', 'Score': gbc_score},
         {'Model':'Gaussian Naive Bayes', 'Score': nb_score},
         {'Model':'Decision Tree', 'Score': dt_score},
         {'Model':'Decision Tree (ANOVA)', 'Score': dt_score_anova},
         {'Model':'Multilayer Perceptron', 'Score': mlp_score},
         {'Model':'Multilayer Perceptron (ANOVA)', 'Score': mlp_score_anova},]
print(pd.DataFrame(score, columns=['Model','Score']))