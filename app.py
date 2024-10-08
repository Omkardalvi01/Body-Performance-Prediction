import streamlit as st
import pickle
import pandas as pd

pickle_in = open('model.pkl','rb')
model,scaler,one_hot,lbl_encoder = pickle.load(pickle_in)

def prediction(dataframe):
    result = model.predict(dataframe)
    return result

def main():
    st.title('Body Performace prediction')
    age = st.number_input('Age:',min_value=0, value=None)
    gender = st.selectbox('Gender:', options=['M','F'])
    height = st.number_input('Height in cm',min_value=0, value=None)
    weight = st.number_input('Weight in KG', min_value=0.0, value=None)
    bf = st.number_input('Body fat in percentage', min_value=0.0, value=None)
    diaolistic = st.number_input('Diastolic', min_value=0.0 , value=None)
    systolic = st.number_input('Systolic', min_value=0.0, value=None)
    gripforce = st.number_input('Grip force in Kg', min_value=0.0, value=None)
    sb_cm = st.number_input('Sit and bend in cm', min_value=0.0, value=None)
    situp = st.number_input('Sit up counts', min_value=0, value=None)
    b_jump = st.number_input('Broad jump in cm', min_value=0, value=None)

    data = pd.DataFrame({
        'age' : [age], 'gender' : [gender], 'height_cm' : [height], 'weight_kg' : [weight], 'body fat_%' : [bf], 'diastolic' : [diaolistic],
       'systolic' : [systolic], 'gripForce' : [gripforce], 'sit and bend forward_cm' : [sb_cm], 'sit-ups counts' : [situp],
       'broad jump_cm' : [b_jump]
    })
    one_hot_values = one_hot.transform(data[['gender']])
    one_hot_df = pd.DataFrame(one_hot_values, columns=one_hot.get_feature_names_out(['gender']))
    data = pd.concat([data.reset_index(drop=True), one_hot_df.reset_index(drop=True)],axis=1)
    data.drop(columns=['gender'],inplace=True)

    scaled_data = scaler.transform(data)

    if st.button("Predict"):
        result = prediction(scaled_data)
        class_of_performance = f'Class: {lbl_encoder.inverse_transform(result)}'
        st.write(class_of_performance)

if __name__ == '__main__':
    main()
    



