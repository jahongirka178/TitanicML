import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title='🚢 Titanic Classifier', layout='wide')
st.title("🚢 Датасет Titanic - Обучение и предсказание")
st.header('Работа с датасетом Titanic')

# Загрузка данных
url = "https://raw.githubusercontent.com/jahongirka178/TitanicML/refs/heads/master/data/titanic_for_hw.csv"
df = pd.read_csv(url)

# Таблица
st.subheader('Данные')
st.dataframe(df.round(2), use_container_width=True)


# Визуализация
st.write('## Визуализация')
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Survived', color='Sex', barmode='group', title='Выжившие по полу')
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.box(df, x='Pclass', y='Age', color='Survived', title='Возраст по классам и выживанию')
    st.plotly_chart(fig2, use_container_width=True)



# Моделирование
X = df.drop(columns=['Survived', 'Name', 'Cabin'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

encoder = ce.TargetEncoder(cols=['Sex', 'Embarked', 'Title', 'FareCategory', 'AgeGroup'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(4),
}

results = []
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
    acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write('## Сравнение моделей по точности')
st.table(pd.DataFrame(results))

# Sidebar для ввода пользователя
st.sidebar.header('Предсказание по параметрам')

sex_input = st.sidebar.selectbox('Пол', df['Sex'].unique())
embarked_input = st.sidebar.selectbox('Порт посадки', df['Embarked'].unique())
title_input = st.sidebar.selectbox('Обращение', df['Title'].unique())
fare_cat_input = st.sidebar.selectbox('Категория тарифа', df['FareCategory'].unique())
age_group_input = st.sidebar.selectbox('Возрастная группа', df['AgeGroup'].unique())

pclass = st.sidebar.selectbox('Класс билета', sorted(df['Pclass'].unique()))
age = st.sidebar.slider('Возраст', float(df['Age'].min()), float(df['Age'].max()), float((df['Age'].min()+df['Age'].max())/2))
fare = st.sidebar.slider('Стоимость билета', float(df['Fare'].min()), float(df['Fare'].max()), float((df['Fare'].min()+df['Fare'].max())/2))
family_size = st.sidebar.slider('Размер семьи', 0, int(df['family_size'].max()), 1)
is_alone = int(family_size == 0)

user_input = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_input,
    'Age': age,
    'Fare': fare,
    'Embarked': embarked_input,
    'Title': title_input,
    'FareCategory': fare_cat_input,
    'family_size': family_size,
    'is_alone': is_alone,
    'AgeGroup': age_group_input
}])

user_encoded = encoder.transform(user_input)

for col in ['Pclass', 'Age', 'Fare', 'family_size', 'is_alone']:
    user_encoded[col] = user_input[col].values

user_encoded = user_encoded[X_train_encoded.columns]

st.dataframe(user_input, use_container_width=True)

st.sidebar.subheader("📈 Результаты предсказания")
for name, model in models.items():
    pred = model.predict(user_encoded)[0]
    proba = model.predict_proba(user_encoded)[0]
    st.sidebar.markdown(f"**{name}: {'Выжил' if pred == 1 else 'Не выжил'}**")
    proba_df = pd.DataFrame({'Класс': ['Не выжил', 'Выжил'], 'Вероятность': proba})
    st.sidebar.dataframe(proba_df.set_index("Класс"), use_container_width=True)
