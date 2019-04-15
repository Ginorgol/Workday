from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from random import randint
import datetime



app = Flask(__name__)
Bootstrap(app)



# import pandas as pd



@app.route('/')
def home():
	return render_template('index.html')



@app.route('/results',methods = ['POST'])
def result():
	if request.method == 'POST':
		description = request.form['description']
		curr_day =  datetime.date.today()
		curr_day='{:%d/%m/%Y}'.format(curr_day)
		incident_number=randint(1000000,9999999)
		creation_date = curr_day
		assigned_group="CallScript"
		product="CallScript"



	# Python model code here

	return render_template("output.html", incident_number = incident_number, creation_date = creation_date, description = description, assigned_group=assigned_group, product = product)



@app.errorhandler(405)
def page_not_found(error):
	return redirect('http://localhost:5000/')



if __name__=='__main__':
	app.run()