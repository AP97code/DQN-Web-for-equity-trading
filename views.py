from flask import Flask
from flask import render_template
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
import yfinance as yf
from datetime import datetime
from . import app
import pandas
from pandas import DataFrame
from .AP_DQN_web import plot_stock_data, run_many_train_test, clean_data, \
        plot_model, ContinuousOHLCVEnv, QNetwork, train_dqn, test_dqn ,calculate_trading_statistics
import os
import json
import quantstats as qs

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/run_model", methods= ["GET", "POST"])
def run_model():
    # plot_encoded_dict = {}
    # trading_statistics_full = {}
    if request.method == "POST":
        ticker = request.form["ticker"]
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        data_split = int(request.form['data_split'])

        ohlcv_data= yf.download(ticker, start=start_date,
                                end=end_date)
        directory = os.getcwd()
        directory_input = os.path.join(directory, "Input")
        #directory = r"C:\Users\augus\OneDrive\Desktop\Stockholm University\Nomades\Python\DQN\Input"
        file_name = f"{ticker}_daily.csv"
        file_path = os.path.join(directory_input, file_name)
        ohlcv_data.to_csv(file_path)
        
        ohlcv_data, ohlcv_data_train, ohlcv_data_test = clean_data(file_path, data_split) #cleaning data and prep for model
        
        test_runs = {
            'run_1':json.loads(request.form['dqn_params_run_1']),
            'run_2':json.loads(request.form['dqn_params_run_2']),
            'run_3': json.loads(request.form['dqn_params_run_3'])
        }
        data_from_tests, results = run_many_train_test(test_runs, ohlcv_data_train, ohlcv_data_test)
        
        plot_for_stock_data = plot_stock_data(ohlcv_data, ticker, data_split)

        plot_encoded_dict = {}
        
        for run_name, run_results in results.items():
            final_returns = run_results['train']
            final_returns_test = run_results['test']
            plot_encoded_dict[run_name] = plot_model(final_returns, final_returns_test, 
                                                     step_data_filename=f"test_step_data_model_{run_name}.csv") 
        
        directory_for_test = os.path.join(directory, "DQN")
        file_names = ["test_step_data_model_run_1.csv", 
                      "test_step_data_model_run_2.csv", 
                      "test_step_data_model_run_3.csv"]

        trading_statistics_full = []
        for file_name in file_names:
            full_file_path = os.path.join(directory_for_test, file_name)
            if os.path.exists(full_file_path):
                trading_statistics = calculate_trading_statistics(full_file_path,
                                                    file_name.rsplit('.', 1)[0])
                trading_statistics_full.append(trading_statistics)
            else:
                print(f"File not found: {full_file_path}")
        return render_template("home.html", 
                               ticker = ticker,
                               plot_encoded_dict=plot_encoded_dict,
                               data_from_tests=data_from_tests,
                               plot_for_stock_data = plot_for_stock_data,
                               trading_statistics=trading_statistics_full)
    else:
        return render_template("home.html")

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")