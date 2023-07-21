import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('train.csv')


df.drop(['id','career_end','career_start','city','people_main','life_main','relation','graduation','langs','has_mobile','followers_count','has_photo', 'bdate', 'occupation_name', 'last_seen'], axis=1, inplace=True)
#df.info()
df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'],axis = 1, inplace = True)
#df.info()

def sex_apply(sex):
    if sex == 2:
        return 0
    return 1


df['sex'] = df['sex'].apply(sex_apply)
#print(df['sex'].value_counts())

#print(df['education_status'].value_counts())

def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == "Student (Specialist)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1 
    elif edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    elif edu_status == "Candidate of Sciences":
        return 3
    else:
        return 4

df['education_status'] = df['education_status'].apply(edu_status_apply)
#print(df['education_status'].value_counts())


def ocu_type_apply(ocu_type):
    if ocu_type == 'university':
        return 0
    return 1

df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply)

#print(df['occupation_type'].value_counts())

X = df.drop('result', axis=1)
y = df['result']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(random_state = 42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)


print("Точность модели:", accuracy)
