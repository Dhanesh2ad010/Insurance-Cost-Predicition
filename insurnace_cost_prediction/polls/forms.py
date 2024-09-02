from django import forms

class InsuranceForm(forms.Form):
    AGE_CHOICES = [(i, i) for i in range(1, 101)]
    SEX_CHOICES = [('male', 'Male'), ('female', 'Female')]
    SMOKER_CHOICES = [('yes', 'Yes'), ('no', 'No')]
    REGION_CHOICES = [
        ('southwest', 'Southwest'),
        ('southeast', 'Southeast'),
        ('northwest', 'Northwest'),
        ('northeast', 'Northeast')
    ]

    age = forms.ChoiceField(choices=AGE_CHOICES, label='Enter Age', widget=forms.Select(attrs={'class': 'form-control'}))
    sex = forms.ChoiceField(choices=SEX_CHOICES, label='Enter Sex', widget=forms.Select(attrs={'class': 'form-control'}))
    bmi = forms.FloatField(label='Enter BMI', min_value=1, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    children = forms.IntegerField(label='Enter Children', min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    smoker = forms.ChoiceField(choices=SMOKER_CHOICES, label='Enter Smoker', widget=forms.Select(attrs={'class': 'form-control'}))
    region = forms.ChoiceField(choices=REGION_CHOICES, label='Enter Region', widget=forms.Select(attrs={'class': 'form-control'}))
