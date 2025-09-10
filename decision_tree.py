import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv("loan_data_set.csv")
df.head()
print(df.shape)

df = df.drop(['Loan_ID'], axis = 1)

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

df = pd.get_dummies(df)

# Drop columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)

df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)

sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange')

X = df.drop(["Loan_Status"], axis=1)
feature_cols = X.columns.tolist()   # ใช้จัดคอลัมน์ให้ตรงตอนพยากรณ์

y = df["Loan_Status"]

# ===== แก้ตำแหน่ง SMOTE: split ก่อน แล้วค่อยทำ SMOTE เฉพาะ train =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# (พล็อตสัดส่วนคลาสบนชุด train หลัง SMOTE ให้หน้าตาคล้ายเดิม)
sns.set_theme(style="darkgrid")
sns.countplot(x=y_train, palette="coolwarm")
plt.ylabel('')
plt.xlabel('Total')
plt.show()

# ===== ต้นไม้ตัดสินใจ: กวาดค่า max_leaf_nodes แบบเดิม =====
scoreListDT = []
for i in range(2,21):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,21), scoreListDT)
plt.xticks(np.arange(2,21,1))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAcc = max(scoreListDT)
print("Decision Tree Accuracy: {:.2f}%".format(DTAcc*100))

# ===== ประเมินผลด้วยรุ่นที่ดีที่สุด (เหมือนบล็อกสรุปของเดิมตอน GB) =====
best_leaf = 2 + int(np.argmax(scoreListDT))
DTclassifier_final = DecisionTreeClassifier(max_leaf_nodes=best_leaf, random_state=0)
DTclassifier_final.fit(X_train, y_train)
y_pred = DTclassifier_final.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# get input# ===== Prediction via Terminal Input =====
def ask_choice(prompt, choices):
    choices_disp = "/".join(choices)
    while True:
        val = input(f"{prompt} ({choices_disp}): ").strip()
        if val in choices:
            return val
        print(f"กรุณากรอกให้ตรงตัวเลือก: {choices_disp}")

def ask_float(prompt):
    while True:
        s = input(f"{prompt}: ").strip()
        try:
            return float(s)
        except ValueError:
            print("กรุณากรอกเป็นตัวเลข")

def build_raw_row():
    # รับอินพุตในรูปแบบ "คอลัมน์ดิบ" ตามชุดข้อมูลเดิม
    row = {
        "Gender":            ask_choice("Gender", ["Male","Female"]),
        "Married":           ask_choice("Married", ["Yes","No"]),
        "Dependents":        ask_choice("Dependents", ["0","1","2","3+"]),
        "Education":         ask_choice("Education", ["Graduate","Not Graduate"]),
        "Self_Employed":     ask_choice("Self_Employed", ["Yes","No"]),
        "ApplicantIncome":   ask_float("ApplicantIncome"),
        "CoapplicantIncome": ask_float("CoapplicantIncome"),
        "LoanAmount":        ask_float("LoanAmount"),
        "Loan_Amount_Term":  ask_float("Loan_Amount_Term"),
        "Credit_History":    ask_float("Credit_History (ใส่ 1 หรือ 0)"),
        "Property_Area":     ask_choice("Property_Area", ["Urban","Semiurban","Rural"])
    }
    return pd.DataFrame([row])

def preprocess_single(raw_df):
    """
    ทำขั้นตอนเดียวกับตอนเทรน:
    - get_dummies
    - drop คอลัมน์ dummy ด้านหนึ่ง
    - rename ให้ชื่อสั้น
    - sqrt กับคอลัมน์เชิงปริมาณที่เราใช้ (ตามโค้ดเดิม)
    - จัดคอลัมน์ให้ตรงกับ feature_cols (ถ้าไม่มี ให้เติม 0)
    """
    # one-hot
    d = pd.get_dummies(raw_df, drop_first=False)

    # drop ด้านหนึ่งของคู่ dummy เหมือนตอนเทรน
    cols_to_drop = ['Gender_Female', 'Married_No', 'Education_Not Graduate',
                    'Self_Employed_No', 'Loan_Status_N']  # อันสุดท้ายไม่มีในอินพุต แต่ใส่ไว้ไม่เป็นไร
    d = d.drop(columns=[c for c in cols_to_drop if c in d.columns], errors="ignore")

    # rename ให้ตรงกับตอนเทรน
    rename_map = {
        'Gender_Male': 'Gender',
        'Married_Yes': 'Married',
        'Education_Graduate': 'Education',
        'Self_Employed_Yes': 'Self_Employed',
        'Loan_Status_Y': 'Loan_Status'  # ไม่มีในอินพุต แต่ไม่เป็นไร
    }
    d.rename(columns=rename_map, inplace=True)

    # sqrt เหมือนเดิม (clip กันค่าติดลบ)
    for col in ["ApplicantIncome","CoapplicantIncome","LoanAmount"]:
        if col in d.columns:
            d[col] = np.sqrt(pd.to_numeric(d[col], errors="coerce").fillna(0).clip(lower=0))

    # จัดคอลัมน์ให้ “ครบและเรียงตรง” กับตอนเทรน
    d = d.reindex(columns=feature_cols, fill_value=0)

    return d

def predict_once():
    raw = build_raw_row()
    X_one = preprocess_single(raw)
    y_hat = DTclassifier_final.predict(X_one)[0]
    # ถ้าอยากได้ความน่าจะเป็นด้วย (ถ้าเปิด predict_proba ได้)
    proba_text = ""
    if hasattr(DTclassifier_final, "predict_proba"):
        p = DTclassifier_final.predict_proba(X_one)[0]
        proba_text = f" | P(Not Approved)={p[0]:.3f}, P(Approved)={p[1]:.3f}"
    label = "Approved" if y_hat==1 else "Not Approved"
    print(f"\nผลทำนาย: {label}{proba_text}\n")

# วนรับหลายรายการได้
if __name__ == "__main__":
    while True:
        predict_once()
        again = input("ทำนายอีกครั้งหรือไม่? (y/n): ").strip().lower()
        if again != "y":
            break


# ===== Visualization: Plot Decision Tree ====

# ====== สร้างชื่อฟีเจอร์แบบย่อ/อ่านง่าย ======
pretty_names = []
for col in X.columns:
    name = col
    # ย่อ prefix ที่ยาวให้สั้นลง (ปรับได้ตามชุดคอลัมน์ของคุณ)
    name = name.replace("Property_Area_", "Area:")
    name = name.replace("Education_", "Edu:")
    name = name.replace("Self_Employed_", "SelfEmp:")
    name = name.replace("Married_", "Married:")
    name = name.replace("Gender_", "Gender:")
    # ถ้าชื่อยังยาวมาก ให้ตัดบรรทัดช่วยอ่าน
    if len(name) > 16:
        name = name.replace("_", " ")
        mid = len(name)//2
        name = name[:mid] + "\n" + name[mid:]
    pretty_names.append(name)

# ====== วาดเฉพาะ Top-k ชั้น เพื่อให้ไม่รก ======
plt.figure(figsize=(16, 9), dpi=150)
tree.plot_tree(
    DTclassifier_final,
    feature_names=pretty_names,                 # ใช้ชื่อแบบย่อ
    class_names=['Not Approved', 'Approved'],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3,                               # <<< แสดงแค่ 3 ชั้นบน (ปรับเป็น 2-4 ได้)
    impurity=False,                            # ตัด gini/entropy ออกให้กล่องไม่ยาว
    proportion=True,                           # แสดงสัดส่วนแทนจำนวนดิบ (อ่านง่าย)
    precision=2                                # ปัดทศนิยมให้สั้น
)
plt.title("Decision Tree (Top 3 Levels)")
plt.tight_layout()
plt.show()

# ====== (ทางเลือก) บันทึกไฟล์ความละเอียดสูง/แบบ SVG ไว้ซูมดูทั้งต้น ======
fig = plt.figure(figsize=(28, 16), dpi=200)
tree.plot_tree(
    DTclassifier_final,
    feature_names=pretty_names,
    class_names=['Not Approved', 'Approved'],
    filled=True, rounded=True, fontsize=7,
    max_depth=None,                            # ทั้งต้น (อาจใหญ่) แต่เซฟเป็น SVG จะซูมได้ชัด
    impurity=False, proportion=True, precision=2
)
plt.title("Decision Tree (Full)")
plt.tight_layout()
fig.savefig("decision_tree_full.svg", format="svg")
fig.savefig("decision_tree_full.png", dpi=220)
plt.close(fig)
print("Saved: decision_tree_full.svg / decision_tree_full.png")

