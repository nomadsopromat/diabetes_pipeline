import streamlit as st
import pandas as pd
import joblib

# Загружаем модель (кэшируем, чтобы не загружать заново при каждом действии пользователя)
@st.cache_resource
def load_model():
    # ИЗМЕНЕНО: теперь загружаем легкую модель LightGBM!
    return joblib.load('lgbm_balanced_pipeline.joblib')

model = load_model()

# Настройка страницы
st.set_page_config(page_title="Калькулятор риска диабета", page_icon="🏥")

st.title('🏥 Предсказание риска диабета')
st.write('Заполните данные пациента ниже, чтобы оценить вероятность наличия диабета с помощью Machine Learning.')

# Создаем колонки для красивой верстки
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Пол (gender)', ['Female', 'Male', 'Other'])
    age = st.number_input('Возраст (age)', min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox('Гипертония (hypertension)', [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")
    heart_disease = st.selectbox('Болезни сердца (heart_disease)', [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")

with col2:
    smoking_history = st.selectbox('История курения (smoking_history)',
                                   ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
    bmi = st.number_input('Индекс массы тела (bmi)', min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    hba1c = st.number_input('Уровень HbA1c (HbA1c_level)', min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glucose = st.number_input('Уровень глюкозы (blood_glucose_level)', min_value=50, max_value=400, value=100, step=1)

# Кнопка для предсказания
if st.button('Сделать предсказание', type='primary'):
    # Собираем данные в DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }])

    # Делаем прогноз
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Результат:")

    if prediction == 1:
        st.error(f'⚠️ **Высокий риск диабета!** Вероятность: {probability:.1%}')
    else:
        st.success(f'✅ **Низкий риск диабета.** Вероятность: {probability:.1%}')
