# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:59:25 2015

@author: Dan
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('first_app.html')


if __name__ == '__main__':
    app.run()