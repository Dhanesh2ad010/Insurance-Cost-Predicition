# Generated by Django 5.0.6 on 2024-07-04 14:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0004_remove_policy_doc_no_remove_policy_fraud_flag_and_more'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Policy',
        ),
    ]
