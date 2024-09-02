from django.shortcuts import render
from django.http import JsonResponse
import joblib
import numpy as np
from .forms import InsuranceForm

# Load your trained models
premium_model = joblib.load('premium_model.pkl')
risk_model = joblib.load('high_risk_model.pkl')
fraud_model = joblib.load('fraud_model.pkl')
result=[]

def predict_insurance_cost(request):
    global response_data
    result.clear()
    if request.method == 'POST':
        form = InsuranceForm(request.POST)
        if form.is_valid():
            age = int(form.cleaned_data['age'])
            sex = form.cleaned_data['sex']
            bmi = float(form.cleaned_data['bmi'])
            children = int(form.cleaned_data['children'])
            smoker = form.cleaned_data['smoker']
            region = form.cleaned_data['region']


            sex_map = {'male': 0, 'female': 1}
            smoker_map = {'yes': 1, 'no': 0}
            region_map = {'southwest': 0, 'southeast': 1, 'northeast': 2, 'northwest': 3}

            sex = sex_map[sex]
            smoker = smoker_map[smoker]
            region = region_map[region]
            print(region)
            input_data = [[age, sex, bmi, children, smoker]]
            print(input_data[0])
            def region_data(region,input_data):
                print(region)
                for i in range(0,4):
                    print(i)
                    if region==i:
                        print("check")
                        input_data[0].append(1)
                    else:
                        input_data[0].append(0)
            region_data(region,input_data)

            print(input_data)
           
            print("started....")
            predicted_premium = premium_model.predict(input_data)[0]
            print("running.....")

            risk_category = risk_model.predict(input_data)[0]
            print("running.....")
            if risk_category ==2:
                risk_category="High"
            elif risk_category ==1:
                risk_category="Medium"
            else:
                risk_category="Low"
            fraud_flag = fraud_model.predict(input_data)[0]
            if fraud_flag ==1:
                fraud_flag="True"
            else:
                fraud_flag="False"
            print("completed")
            print(risk_category,predicted_premium,fraud_flag)
            result.append(float(predicted_premium))
            result.append(risk_category)
            result.append(fraud_flag)
            response_data = {
                'predicted_premium': float(predicted_premium),
                'risk_category': risk_category,
                'fraud_flag':fraud_flag
            }
            if 'generate_pdf' in request.POST:
                return generate_pdf(request, response_data)
            else:
                return JsonResponse(response_data)
    else:
        form = InsuranceForm()

    return render(request, 'templates/index.html', {'form': form})

print(result,"result")
from django.shortcuts import render

def home(request):
    return render(request, 'templates/home.html')
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from io import BytesIO 
import os
from datetime import datetime
from reportlab.pdfgen import canvas
def generate_pdf(request):


    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    page_width = 210  # in mm for A4
    page_height = 297  # in mm for A4

    # Define your string positions
    footer_height = 20  # Height of the footer from the bottom in mm
    text_height = page_height - footer_height
    width, height = A4
    generated_date = datetime.now()
    generated_date_str = generated_date.strftime('%Y-%m-%d %H:%M')

    p.setFont("Helvetica-Oblique", 10)
  # Position Y for the text, adjusted for the footer

# Draw your strings
    p.drawString(30, text_height, "Created by Prediction System")
    p.drawString(220, text_height, generated_date_str)
    p.drawString(420, 820, "DATE:")
    p.drawString(460, 820, generated_date_str)
    p.setFont("Times-Roman", 20)
    p.drawString(100, 750, f"Predicted Premium:$ {result[0]}")
    p.drawString(100, 730, f"Risk Category: {result[1]}")
    p.drawString(100, 710, f"Fraud Flag: {result[2]}")
    risk_explanations = {
            'Low': [
                "Low BMI and Non-Smoker: Individuals with BMI below 30 and who do not smoke.",
                "Younger Age and No Children: Younger individuals without children.",
                "Residence in Low-Risk Regions: Living in regions associated with lower high-risk factors."
            ],
            'Medium': [
                "Moderate BMI and Smoking Status: Moderate BMI and smoking status that does not strongly indicate high risk.",
                "Mixed Age and Family Status: Mixed demographics (age and children) without extreme values."
            ],
            'High': [
                "High BMI and Smoker: BMI above 30 and current smoker.",
                "Older Age with Health Risks: Older individuals with health conditions indicated by BMI and smoking.",
                "Residence in High-Risk Regions: Living in regions with higher prevalence of high-risk factors."
            ]
        }

        # Position for risk category explanations
    explanation_y = 630

        # Draw explanations for all risk categories
    p.setFont("Helvetica", 10)
    p.drawString(50, explanation_y, "Reasons for Risk Categories:")

    for category, reasons in risk_explanations.items():
        explanation_y -= 20  # Move up for the next category title
        p.setFont("Helvetica-Bold", 12)
        p.drawString(70, explanation_y, f"{category} Risk:")
            
        for reason in reasons:
            explanation_y -= 15  # Move up for each reason
            p.setFont("Helvetica", 10)
            p.drawString(90, explanation_y, reason)

    p.showPage()
    p.save()

    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')
