# Generated by Django 5.0.6 on 2024-07-02 12:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='policy',
            name='fraud_flag',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AddField(
            model_name='policy',
            name='predicted_premium',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AddField(
            model_name='policy',
            name='risk_category',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
    ]
