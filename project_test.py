# importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib

# import file
df = pd.read_csv('D:\child_nutrition_data.csv')

# แสดงหัวตารางข้อมูลแรก
# แสดงหัวตารางข้อมูลท้าย
# แสดงรูปร่างของข้อมูล (จำนวนแถว, จำนวนคอลัมน์)
df.head()
print('Head\n', df.head())
print('Tail\n', df.tail())
print('Shape\n', df.shape)

# Generating descriptive statistics. แสดงค่าสถิติทางคณิตศาสตร์ของข้อมูล
# print('describe\n', df.describe().T)



# 1. เพศ (Gender)
df['sex_label'] = df['Gender'].map({0: 'Male', 1: 'Female'})
# นับจำนวนผู้ชาย/ผู้หญิง
gender_counts = df['sex_label'].value_counts()
print('Gender counts:\n', gender_counts)
# เตรียมข้อมูลสำหรับกราฟ
labels = gender_counts.index.tolist()
sizes = gender_counts.values
color_map = {'Male': 'steelblue', 'Female': 'lightpink'}
colors = [color_map.get(label, 'grey') for label in labels]
# ฟังก์ชันแสดงทั้ง % และ จำนวนจริง
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({count})'
    return my_autopct
# พล็อต Pie chart
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
        shadow=True, startangle=90, colors=colors)
ax1.axis('equal')
plt.title('Gender Distribution')
plt.show()



# 2. อายุ (Age in months)
age_counts = df['Age/month'].value_counts().sort_index()
# พล็อต bar chart
plt.figure(figsize=(12, 6))
plt.bar(age_counts.index, age_counts.values, color='skyblue')
plt.title('Age Distribution (in months)')
plt.xlabel('Age (months)')
plt.ylabel('Number of Children')
plt.xticks(rotation=90)  # หมุนแกน x ถ้าอายุมาก
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 3. น้ำหนักแรกคลอด (Birth Weight)
plt.figure(figsize=(10, 6))
plt.hist(df['Birth Weight (grams)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Birth Weight')
plt.xlabel('Birth Weight (grams)')
plt.ylabel('Number of Children')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 4. ประวัติการฉีดวัคซีน (Vaccine)
vaccine_columns = [col for col in df.columns if col.startswith('Vaccine')]
# จำนวนเด็กที่ฉีดแล้วในแต่ละประเภท
vaccine_counts = df[vaccine_columns].sum().sort_values(ascending=False)
vaccine_percent = df[vaccine_columns].mean() * 100
#กราฟแบบเปอร์เซ็นต์
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(vaccine_percent.index, vaccine_percent.values, color='skyblue', edgecolor='black')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Vaccination Coverage by Type (%)')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(vaccine_percent.index, rotation=45, ha='right')
ax.set_ylim(0, 110)  # เผื่อที่วางข้อความ
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()



# 5. ภาวะขาดออกซิเจนขณะคลอด (Birth Asphyxia Status)
# แปลงค่าตัวเลขเป็นข้อความเพื่อความเข้าใจง่าย
df['Birth Asphyxia Label'] = df['Birth Asphyxia Status'].map({0: 'No Asphyxia', 1: 'Asphyxia'})
# นับจำนวนแต่ละกลุ่ม
asphyxia_counts = df['Birth Asphyxia Label'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(asphyxia_counts.values, labels=asphyxia_counts.index,
        autopct='%1.1f%%', colors=['#ACE1AF', 'salmon'], startangle=90, shadow=True)
plt.title('Birth Asphyxia Status (%)')
plt.axis('equal')
plt.show()



# 6. การตรวจไทรอยด์ (Thyroid Screening Result)
# แปลงค่าตัวเลขเป็นข้อความ
df['Thyroid Result Label'] = df['Thyroid Screening Result'].map({
    1: 'Abnormal',
    0: 'Normal'
})
# นับจำนวนผลลัพธ์แต่ละประเภท
thyroid_counts = df['Thyroid Result Label'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(thyroid_counts.values, labels=thyroid_counts.index,
        autopct='%1.1f%%', colors=['#c1f0b2', 'gray'], startangle=90, shadow=True)
plt.title('Thyroid Screening Result (%)')
plt.axis('equal')
plt.show()



# 7. ภาวะแทรกซ้อนขณะตั้งครรภ์ (Pregnancy Complications)
# นับจำนวนแต่ละประเภท
pregnancy_counts = df['Pregnancy Complications'].value_counts()
# วาด Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    pregnancy_counts.values,
    labels=pregnancy_counts.index,
    autopct='%1.1f%%',
    colors=['#fbfdaa', 'salmon', 'gray'],
    startangle=90,
    shadow=True
)
plt.title('Pregnancy Complications (%)')
plt.axis('equal')
plt.show()



# 8. การดื่มนม (Breastfeeding Duration)
# แปลงค่าตัวเลขให้เป็นข้อความ (Label)
df['Breastfeeding Label'] = df['Breastfeeding Duration'].map({
    0: 'No Breastfeeding',
    1: '3 Months',
    2: '6 Months'
}).fillna('Unknown')
# นับจำนวนในแต่ละกลุ่ม
bf_counts = df['Breastfeeding Label'].value_counts().reindex(['No Breastfeeding', '3 Months', '6 Months'])
# ฟังก์ชันแสดง % + จำนวนจริง
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({count})'
    return my_autopct
# วาด Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    bf_counts.values,
    labels=bf_counts.index,
    autopct=make_autopct(bf_counts.values),
    colors=['#F7CAC9', '#A8D5BA', '#92D4A8'],
    startangle=90,
    shadow=True
)
plt.title('Breastfeeding Duration (%) and Count')
plt.axis('equal')
plt.tight_layout()
plt.show()



# 9. ลำดับการเกิด (Birth Order)
# นับจำนวนเด็กตามลำดับการเกิด (เรียงจาก 1 ขึ้นไป)
birth_order_counts = df['Birth Order'].value_counts().sort_index()
# วาดกราฟแท่ง
plt.figure(figsize=(8, 5))
bars = plt.bar(birth_order_counts.index.astype(str), birth_order_counts.values, color='#A8D5BA')
# ใส่จำนวนบนแท่ง
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom')
plt.title('Distribution of Birth Order')
plt.xlabel('Birth Order')
plt.ylabel('Number of Children')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# 10. อายุบิดา/มารดา (Father’s/Mother’s Age)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 1. Father's Age Histogram
n_father, bins_father, patches_father = axes[0].hist(
    df["Father's Age"].dropna(), bins=15, color='#9EC6F3', edgecolor='black'
)
axes[0].set_title("Father's Age Distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Number of Fathers")
# ใส่จำนวนบน bar
for i in range(len(patches_father)):
    height = n_father[i]
    bin_center = (bins_father[i] + bins_father[i + 1]) / 2
    axes[0].text(bin_center, height + 1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)
# 2. Mother's Age Histogram
n_mother, bins_mother, patches_mother = axes[1].hist(
    df["Mother's Age"].dropna(), bins=15, color='#F7CAC9', edgecolor='black'
)
axes[1].set_title("Mother's Age Distribution")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Number of Mothers")
# ใส่จำนวนบน bar
for i in range(len(patches_mother)):
    height = n_mother[i]
    bin_center = (bins_mother[i] + bins_mother[i + 1]) / 2
    axes[1].text(bin_center, height + 1, f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()



# 11. อาชีพบิดา/มารดา (Father’s/Mother’s Occupation)
# ตั้งค่าฟอนต์รองรับภาษาไทย
matplotlib.rcParams['font.family'] = 'Tahoma'
# Mapping EN+TH
occupation_map_combined = {
    0: 'Agriculture (เกษตรกรรม)',
    1: 'Trader (ค้าขาย)',
    2: 'Business Owner (เจ้าของกิจการ)',
    3: 'Technician/Engineer (ช่างเทคนิค/วิศวกรรม)',
    4: 'Police Officer (ตำรวจ)',
    5: 'Farmer (ทำฟาร์ม)',
    6: 'Field Worker (ทำไร่)',
    7: 'Company Employee (พนักงานบริษัท)',
    8: 'Daily Laborer (รับจ้าง)',
    9: 'Housewife (แม่บ้าน)',
    10: 'Government Officer (รับราชการ)'
}
# แปลงรหัสอาชีพ
df['Father Occupation'] = df["Father's Occupation"].map(occupation_map_combined)
df['Mother Occupation'] = df["Mother's Occupation"].map(occupation_map_combined)
# --- กราฟอาชีพบิดา ---
father_counts = df['Father Occupation'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
bars1 = plt.bar(father_counts.index, father_counts.values, color='#9EC6F3')
plt.title("Father's Occupation")
plt.ylabel('Number of Fathers')
plt.xticks(rotation=45, ha='right')
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()
# --- กราฟอาชีพมารดา ---
mother_counts = df['Mother Occupation'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
bars2 = plt.bar(mother_counts.index, mother_counts.values, color='#F7CAC9')
plt.title("Mother's Occupation")
plt.ylabel('Number of Mothers')
plt.xticks(rotation=45, ha='right')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()



# 12. การสูบบุหรี่ของบิดา/มารดา (Father’s/Mother’s Smoking Status)
# Mapping การสูบบุหรี่
smoking_map = {
    0: 'Non-Smoker',
    1: 'Smoker'
}
# แปลงข้อมูล
df['Father Smoking Status'] = df["Father's Smoking Status"].map(smoking_map).fillna('Unknown')
df['Mother Smoking Status'] = df["Mother's Smoking Status"].map(smoking_map).fillna('Unknown')
smoking_color_map = {
    'Smoker': '#FF9999',
    'Non-Smoker': '#A8D5BA',
}
# -------- กราฟพ่อ --------
father_smoke_counts = df['Father Smoking Status'].value_counts()
colors_father = [smoking_color_map.get(label, '#BBBBBB') for label in father_smoke_counts.index]
plt.figure(figsize=(6, 5))
bars1 = plt.bar(father_smoke_counts.index, father_smoke_counts.values, color=colors_father)
plt.title("Father's Smoking Status")
plt.ylabel('Number of Fathers')
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
# -------- กราฟแม่ --------
mother_smoke_counts = df['Mother Smoking Status'].value_counts()
colors_mother = [smoking_color_map.get(label, '#BBBBBB') for label in mother_smoke_counts.index]
plt.figure(figsize=(6, 5))
bars2 = plt.bar(mother_smoke_counts.index, mother_smoke_counts.values, color=colors_mother)
plt.title("Mother's Smoking Status")
plt.ylabel('Number of Mothers')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()



# 13.การดื่มแอลกอฮอล์ของบิดา/มารดา (Father’s/Mother’s Alcohol Consumption)
# Mapping สำหรับการดื่มแอลกอฮอล์
alcohol_map = {
    0: 'Non-Drinker',
    1: 'Drinker'
}
# แปลงค่าจากรหัสเป็นข้อความ
df['Father Alcohol Status'] = df["Father's Alcohol Consumption"].map(alcohol_map).fillna('Unknown')
df['Mother Alcohol Status'] = df["Mother's Alcohol Consumption"].map(alcohol_map).fillna('Unknown')
# สีสำหรับแต่ละกลุ่ม
alcohol_color_map = {
    'Drinker': '#FFCC99',       # ส้มพาสเทล
    'Non-Drinker': '#A8D5BA',   # เขียวพาสเทล
}
# -------- กราฟพ่อ --------
father_alcohol_counts = df['Father Alcohol Status'].value_counts()
colors_father = [alcohol_color_map.get(label, '#BBBBBB') for label in father_alcohol_counts.index]
plt.figure(figsize=(6, 5))
bars1 = plt.bar(father_alcohol_counts.index, father_alcohol_counts.values, color=colors_father)
plt.title("Father's Alcohol Consumption")
plt.ylabel('Number of Fathers')
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
# -------- กราฟแม่ --------
mother_alcohol_counts = df['Mother Alcohol Status'].value_counts()
colors_mother = [alcohol_color_map.get(label, '#BBBBBB') for label in mother_alcohol_counts.index]
plt.figure(figsize=(6, 5))
bars2 = plt.bar(mother_alcohol_counts.index, mother_alcohol_counts.values, color=colors_mother)
plt.title("Mother's Alcohol Consumption")
plt.ylabel('Number of Mothers')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()



# 14. การใช้สารเสพติดของบิดา/มารดา (Father’s/Mother’s Substance Use)
# Mapping ค่าตัวเลขเป็นข้อความ
substance_map = {
    0: 'Non-User',
    1: 'User'
}
# แปลงค่าของพ่อและแม่
df['Father Substance Use'] = df["Father's Substance Use"].map(substance_map).fillna('Unknown')
df['Mother Substance Use'] = df["Mother's Substance Use"].map(substance_map).fillna('Unknown')
# กำหนดสีแยกตามสถานะ
substance_color_map = {
    'User': '#FF9999',
    'Non-User': '#A8D5BA',
}
# -------- กราฟพ่อ --------
father_substance_counts = df['Father Substance Use'].value_counts()
colors_father = [substance_color_map.get(label, '#BBBBBB') for label in father_substance_counts.index]
plt.figure(figsize=(6, 5))
bars1 = plt.bar(father_substance_counts.index, father_substance_counts.values, color=colors_father)
plt.title("Father's Substance Use")
plt.ylabel('Number of Fathers')
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
# -------- กราฟแม่ --------
mother_substance_counts = df['Mother Substance Use'].value_counts()
colors_mother = [substance_color_map.get(label, '#BBBBBB') for label in mother_substance_counts.index]
plt.figure(figsize=(6, 5))
bars2 = plt.bar(mother_substance_counts.index, mother_substance_counts.values, color=colors_mother)
plt.title("Mother's Substance Use")
plt.ylabel('Number of Mothers')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()



# 15. ผู้ที่เด็กอาศัยอยู่ด้วย (Child’s Living Situation)
# ฟอนต์ภาษาไทย
matplotlib.rcParams['font.family'] = 'Tahoma'
# Mapping Living Situation
living_map = {
    0: 'Grandparents/Relatives (ญาติ, ตา-ยาย)',
    1: 'Parents (พ่อแม่)'
}
# แปลงรหัสเป็นข้อความ
df['Living Situation Label'] = df["Child's Living Situation"].map(living_map).fillna('Unknown')
# นับจำนวน
living_counts = df['Living Situation Label'].value_counts()
# กำหนดสีแยกกลุ่ม
color_map = {
    'Parents (พ่อแม่)': '#A8D5BA',          # เขียวพาสเทล
    'Grandparents/Relatives (ญาติ, ตา-ยาย)': '#FFCC99',  # ส้มพาสเทล
    'Unknown': '#CCCCCC'
}
colors = [color_map.get(label, '#BBBBBB') for label in living_counts.index]
# พล็อตกราฟแท่ง
plt.figure(figsize=(8, 5))
bars = plt.bar(living_counts.index, living_counts.values, color=colors)
plt.title("Child's Living Situation")
plt.ylabel("Number of Children")
plt.xticks(rotation=15)
# ใส่จำนวนบนแท่ง
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


