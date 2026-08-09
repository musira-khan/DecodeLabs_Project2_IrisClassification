import pickle
import streamlit as st
import numpy as np

scaler_file = pickle.load(open('scaler.pkl', 'rb'))
model_file = pickle.load(open('model.pkl', 'rb'))


def pred_output(user_input):
    scaled_input = scaler_file.transform(
        np.array(user_input).reshape(1, 4)
    )

    ypred = model_file.predict(scaled_input)

    return ypred[0]


def main():

    st.title('Iris Classification')
    st.write('Predict the Iris flower class using Machine Learning.')

    st.divider()

    sepalLength = st.number_input(
        'Enter the Sepal Length (cm)'
    )

    sepalWidth = st.number_input(
        'Enter the Sepal Width (cm)'
    )

    petalLength = st.number_input(
        'Enter the Petal Length (cm)'
    )

    petalWidth = st.number_input(
        'Enter the Petal Width (cm)'
    )

    if st.button('Predict'):

        user_input = [
            sepalLength,
            sepalWidth,
            petalLength,
            petalWidth
        ]

        prediction = pred_output(user_input)

        st.success(f'Predicted Class: {prediction}')


if __name__ == '__main__':
    main()
