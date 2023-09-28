import pickle
import pandas as pd
import math
import gradio as gr
import pandas as pd


## test

with open('hospitalStay.pkl', 'rb') as f:
    model = pickle.load(f)


default_values = {
    'rcount': 0.0,
    'gender': 0.0,
    'dialysisrenalendstage': False,
    'asthma': False,
    'irondef': False,
    'pneum': False,
    'substancedependence': False,
    'psychologicaldisordermajor': False,
    'depress': False,
    'psychother': False,
    'fibrosisandother': False,
    'malnutrition': False,
    'hemo': 0.0,
    'hematocrit': 11.9,
    'neutrophils': 9.4,
    'sodium': 135.885126,
    'glucose': 23.765383,
    'bloodureanitro': 12.0,
    'creatinine': 0.268453,
    'bmi': 29.798116,
    'pulse': 74.0,
    'respiration': 6.5,
    'secondarydiagnosisnonicd9': 1.0,
    'facid': 4.0
}


def predict_length_of_stay(rcount, gender, asthma, hematocrit, bmi):
    
    # Convert input values from Gradio input
    if rcount < 0 or rcount > 5:
            return "Invalid value for Number of Admissions. Use a number between 0 and 5"
    gender = 1.0 if gender == "Male" else 0.0
    
    input_values = default_values.copy()
    input_values.update({
        'rcount': rcount,
        'gender': gender,
        'asthma': asthma,
        'hematocrit': hematocrit,
        'bmi': bmi
    })
    
    df = pd.DataFrame([input_values])
    
    prediction = model.predict(df)
    rounded_prediction = math.ceil(prediction[0])
    return rounded_prediction


iface = gr.Interface(
    fn=predict_length_of_stay,
    title="Calculate the length of a patient's hospital stay.",
    description="The model takes in 25 variables to predict the length of a hospital stay, but I have limited this demo to only include 5 input values.",
    inputs=[
        gr.components.Number(label="Number of Admissions in the Last 180 Days", default=0.0),
        gr.components.Dropdown(choices=["Male", "Female"], label="Gender", default="Male"),
        gr.components.Checkbox(label="Asthma", default=False),
        gr.components.Number(label="Hematocrit", default=11.9),
        gr.components.Number(label="BMI", default=29.798116),
    ],
    allow_flagging="never",
    outputs=gr.components.Textbox(label="Predicted Length of Stay (days)")
)

if __name__ == '__main__':
    iface.launch()
