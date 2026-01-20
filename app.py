import gradio as gr
import joblib
import numpy as np

# 1. Load the pre-trained model and scaler (the files you uploaded)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_risk(age, income, loan_amount, credit_score):
    features = np.array([[age, income, loan_amount, credit_score]])
    features_scaled = scaler.transform(features)
    
    # Get the probability of default (Class 1)
    # [Probability of Safe, Probability of Default]
    probs = model.predict_proba(features_scaled)[0]
    default_prob = probs[1] 

    # Set a custom threshold (e.g., 0.3 instead of 0.5)
    threshold = 0.3 
    
    if default_prob >= threshold:
        return f"⚠️ High Risk: {round(default_prob*100, 2)}% Default Probability"
    else:
        return f"✅ Low Risk: {round(default_prob*100, 2)}% Default Probability"

# 5. Build UI
interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Credit Score")
    ],
    outputs="text",
    title="AI Credit Risk Predictor"
)

interface.launch()