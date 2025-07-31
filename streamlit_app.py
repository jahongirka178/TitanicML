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
st.title("🚢 Датасет Titanic - Предсказание выживания пассажира")
st.write('## Классическое ML-приложение на Titanic')

# Загрузка данных
url = "https://raw.githubusercontent.com/jahongirka178/TitanicML/refs/heads/master/data/titanic_for_hw.csv"
df = pd.read_csv(url)


# Визуализация
st.subheader('📊 Визуализация данных')

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Survived', color='Sex', barmode='group',
                        title='Выживание по полу', labels={'Survived': 'Выжил'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x='Age', y='Fare', color=df['Survived'].map({0: 'Не выжил', 1: 'Выжил'}),
                      title='Возраст vs Тариф', labels={'Fare': 'Стоимость билета', 'Age': 'Возраст'})
    st.plotly_chart(fig2, use_container_width=True)

# Разделение признаков и цели
X = df.drop(columns='Survived')
y = df['Survived']

# Кодирование категориальных переменных
encoder = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_encoded = encoder.fit_transform(X, y)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Модели
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.subheader('📈 Сравнение моделей')
st.table(pd.DataFrame(results))

# Форма ввода данных
st.sidebar.header('🔍 Предсказание по параметрам')
pclass = st.sidebar.selectbox('Класс билета (Pclass)', sorted(df['Pclass'].unique()))
sex = st.sidebar.selectbox('Пол (Sex)', df['Sex'].unique())
age = st.sidebar.slider('Возраст (Age)', 0, 80, 30)
sibsp = st.sidebar.number_input('Количество братьев/сестёр или супругов (SibSp)', 0, 10, 0)
parch = st.sidebar.number_input('Количество родителей/детей (Parch)', 0, 10, 0)
fare = st.sidebar.slider('Стоимость билета (Fare)', 0.0, 600.0, 50.0)
embarked = st.sidebar.selectbox('Порт посадки (Embarked)', df['Embarked'].unique())

# Подготовка входных данных
user_input = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

user_encoded = encoder.transform(user_input)

st.subheader('Введённые данные')
st.dataframe(user_input)

# Предсказания моделей
st.sidebar.subheader("📌 Результаты предсказания")

if st.sidebar.button("Сделать предсказание"):
    user_encoded = encoder.transform(user_input)

    for name, model in models.items():
        pred = model.predict(user_encoded)[0]
        proba = model.predict_proba(user_encoded)[0]

        st.sidebar.markdown(f"**{name}: {'✅ Выжил' if pred == 1 else '❌ Не выжил'}**")
        proba_df = pd.DataFrame({'Класс': ['Не выжил', 'Выжил'], 'Вероятность': proba})
        st.sidebar.dataframe(proba_df.set_index("Класс"), use_container_width=True)
else:
    st.sidebar.markdown("⬅️ Введите параметры и нажмите кнопку.")