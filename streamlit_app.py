import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import plotly.express as px

pd.set_option("display.float_format", "{:.2f}".format)


def get_fare_category(fare: float) -> str:
    """
    Возвращает категорию тарифа по значению fare,
    используя те же квантильные границы, что и pd.qcut.

    Parameters:
        fare (float): Значение тарифа

    Returns:
        str: Одна из категорий ['Low', 'Medium', 'High', 'VeryHigh']
    """
    quantiles = [0, 0.3, 0.5, 0.85, 1.0]
    labels = ['Low', 'Medium', 'High', 'VeryHigh']
    bins = df['Fare'].quantile(quantiles).values

    # Обрабатываем вручную через pd.cut (одиночный элемент)
    category = pd.cut([fare], bins=bins, labels=labels, include_lowest=True)[0]

    return str(category)


def get_age_group(age: float) -> str:
    """
    Возвращает возрастную категорию на основе значения возраста.

    Parameters:
        age (float): Возраст

    Returns:
        str: Одна из категорий ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    """
    if age <= 12:
        return 'Child'
    elif age <= 19:
        return 'Teen'
    elif age <= 35:
        return 'YoungAdult'
    elif age <= 59:
        return 'Adult'
    else:
        return 'Senior'


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
# Первая строка визуализаций
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Survived', color='Sex', barmode='group',
                        title='Выжившие по полу')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x='Pclass', color='Survived', barmode='group',
                        title='Распределение классов по выжившим')
    st.plotly_chart(fig2, use_container_width=True)

# Вторая строка визуализаций — здесь мы создаём col3 и col4
col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(df, x='Fare', nbins=50, color='Survived',
                        title='Распределение стоимости билета по выжившим')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.histogram(df, x='Age', nbins=40, color='Survived',
                        title='Распределение возраста по выжившим')
    st.plotly_chart(fig4, use_container_width=True)


# Моделирование
X = df.drop(columns=['Survived', 'Name', 'Cabin'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

encoder = ce.TargetEncoder(cols=['Sex', 'Embarked', 'Title', 'FareCategory', 'AgeGroup'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(9)),
    'Logistic Regression': make_pipeline(StandardScaler(),
                                         LogisticRegression(C=0.01, max_iter=1000, penalty='l1', solver='liblinear')),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        criterion='entropy',
        random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    )

}

results = []
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
    acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
    results.append({
        'Model': name,
        'Test Accuracy': acc_test,
        'Train Accuracy': acc_train
    })


st.write('## Сравнение моделей по точности')
st.table(pd.DataFrame(results).round(2))

# Sidebar для ввода пользователя
st.sidebar.header('Предсказание по параметрам')

sex_input = st.sidebar.selectbox('Пол', df['Sex'].unique())
embarked_input = st.sidebar.selectbox('Порт посадки', df['Embarked'].unique())
title_input = st.sidebar.selectbox('Обращение', df['Title'].unique())
pclass = st.sidebar.selectbox('Класс билета', sorted(df['Pclass'].unique()))

age = st.sidebar.slider('Возраст',
                        float(df['Age'].min()),
                        float(df['Age'].max()),
                        float((df['Age'].min() + df['Age'].max()) / 2)
                        )

fare = st.sidebar.slider(
    'Стоимость билета',
    float(df['Fare'].min()),
    float(df['Fare'].max()),
    float((df['Fare'].min() + df['Fare'].max()) / 2)
)

family_size = st.sidebar.slider(
    'Размер семьи',
    0,
    int(df['family_size'].max()),
    int(df['family_size'].max() / 2)
)
is_alone = int(family_size == 0)

user_input = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_input,
    'Age': age,
    'Fare': fare,
    'Embarked': embarked_input,
    'Title': title_input,
    'FareCategory': get_fare_category(fare),
    'family_size': family_size,
    'is_alone': is_alone,
    'AgeGroup': get_age_group(age)
}])

df = df.round(2)

user_encoded = encoder.transform(user_input)

for col in ['Pclass', 'Age', 'Fare', 'family_size', 'is_alone']:
    user_encoded[col] = user_input[col].values

user_encoded = user_encoded[X_train_encoded.columns]

#st.dataframe(user_input, use_container_width=True)

st.sidebar.subheader("📈 Результаты предсказания")

for name, model in models.items():
    pred = model.predict(user_encoded)[0]
    proba = model.predict_proba(user_encoded)[0]
    st.sidebar.markdown(f"**{name}: {'Выжил' if pred == 1 else 'Не выжил'}**")
    proba_df = pd.DataFrame({'Класс': ['Не выжил', 'Выжил'], 'Вероятность': proba})
    st.sidebar.dataframe(proba_df.set_index("Класс"), use_container_width=True)
