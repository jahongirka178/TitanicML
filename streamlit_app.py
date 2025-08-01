import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import plotly.express as px

pd.set_option("display.float_format", "{:.2f}".format)


def get_metrics(y, y_pred, y_proba):
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    return auc(fpr, tpr), accuracy_score(y, y_pred), f1_score(y, y_pred)


def show_roc(y, y_pred, y_proba, title):
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_score = auc(fpr, tpr)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    df_roc = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig = px.line(
        df_roc,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"{title}. AUC={auc_score:.4f}. Accuracy={accuracy * 100:.2f}%. F1={f1:.4f}.",
        markers=True
    )
    # Добавляем диагональ случайного классификатора
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="grey")
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def analyze_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    auc_score_test, accuracy_test, f1_test = get_metrics(y_test, y_pred_test, y_proba_test)
    auc_score_train, accuracy_train, f1_train = get_metrics(y_train, y_pred_train, y_proba_train)

    show_roc(y_test, y_pred_test, y_proba_test, f'{model_name}')

    return {
        'model': model_name,
        'auc_test': auc_score_test,
        'auc_train': auc_score_train,
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'f1_test': f1_test,
        'f1_train': f1_train,
    }


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

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Survived', color='Sex', barmode='group',
                        title='Выжившие по полу')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x='Pclass', color='Survived', barmode='group',
                        title='Распределение классов по выжившим')
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(df, x='Fare', nbins=50, color='Survived',
                        title='Распределение стоимости билета по выжившим', opacity=0.6)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.histogram(df, x='Age', nbins=40, color='Survived',
                        title='Распределение возраста по выжившим', opacity=0.6)
    st.plotly_chart(fig4, use_container_width=True)

# Моделирование
X = df.drop(columns=['Survived', 'Name', 'Cabin'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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

# st.dataframe(user_input, use_container_width=True)

st.sidebar.subheader("📈 Результаты предсказания")

for name, model in models.items():
    y_pred = model.predict(user_encoded)[0]
    y_proba = model.predict_proba(user_encoded)[0]
    st.sidebar.markdown(f"**{name}: {'Выжил' if y_pred == 1 else 'Не выжил'}**")
    proba_df = pd.DataFrame({'Класс': ['Не выжил', 'Выжил'], 'Вероятность': y_proba})
    st.sidebar.dataframe(proba_df.set_index("Класс"), use_container_width=True)

# ROC-AUC с выбором классификатора
st.write("## Анализ моделей")
model_choice = st.selectbox(
    "Выберите модель:",
    list(models.keys()),
    index=0
)

model = models[model_choice]

test_size = st.slider(
    "Доля тестовой выборки",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="Слишком малый или слишком большой размер может повлиять на стабильность стратификации."
)

encoder_options = {
    "OneHotEncoder": ce.OneHotEncoder,
    "OrdinalEncoder": ce.OrdinalEncoder,
    "TargetEncoder": ce.TargetEncoder,
    "CatBoostEncoder": ce.CatBoostEncoder,
    "LeaveOneOutEncoder": ce.LeaveOneOutEncoder,
    "BinaryEncoder": ce.BinaryEncoder,
}

encoder_name = st.selectbox("Выберите encoder", list(encoder_options.keys()), index=2)  # по умолчанию TargetEncoder
EncoderClass = encoder_options[encoder_name]

encoder = EncoderClass(cols=['Sex', 'Embarked', 'Title', 'FareCategory', 'AgeGroup'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

result = pd.DataFrame([analyze_model(X_train_encoded, X_test_encoded, y_train, y_test, model, model_choice)])
st.subheader("Результаты анализа")
st.dataframe(result)

st.write("## Stacking")

encoder_name = st.selectbox("Выберите encoder для Stacking", list(encoder_options.keys()),
                            index=2)  # по умолчанию TargetEncoder
EncoderClass = encoder_options[encoder_name]

encoder = EncoderClass(cols=['Sex', 'Embarked', 'Title', 'FareCategory', 'AgeGroup'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

'''
# 1. выбор трёх базовых моделей
stacking_models = st.multiselect(
    "Выберите ровно 3 модели для стекинга (базовые модели):",
    options=list(models.keys()),
    default=list(models.keys())[:3],  # любые 3 по умолчанию
    help="Базовые (первого уровня) модели"
)

# 2. выбор финальной (мета)-модели
final_model_name = st.selectbox(
    "Выберите финальную модель (мета-модель):",
    options=list(models.keys()),
    index=3,
    help="Эта модель обучается на предсказаниях базовых моделей"
)

'''

stacking_models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
final_model_name = 'KNN'

# 3. кнопка запуска
launch_stacking = st.button("🚀 Запустить Stacking")

# 4. логика запуска
if launch_stacking:
    estimators = [(name, models[name]) for name in stacking_models]
    final_model = models[final_model_name]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_model,
        passthrough=True,  # добавлять ли исходные признаки во второй уровень
        cv=5,
        n_jobs=-1
    )

    stacking_result = analyze_model(
        X_train_encoded, X_test_encoded,
        y_train, y_test,
        stacking_clf, "Stacking"
    )

    st.subheader("Результаты стекинга")
    st.dataframe(pd.DataFrame([stacking_result]).round(3))
