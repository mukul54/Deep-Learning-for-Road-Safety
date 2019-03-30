from django.shortcuts import render, redirect

# Create your views here.

def home(request):
    return render(request, 'hackathon/home.html', {})

def login(request):
    return render(request, 'hackathon/login.html', {})