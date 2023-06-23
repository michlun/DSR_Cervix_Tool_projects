from flask import Flask, render_template, url_for

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/', methods=['GET'])
def index():
    css_url = url_for('static', filename='style.css')
    return render_template('./index.html')


@app.route('/model_1.html')
def model_1():
    return render_template('model_1.html')


@app.route('/model_2.html')
def model_2():
    return render_template('model_2.html')


@app.route('/', methods=['POST'])
def move_to_model_2():
    pass


if __name__ == '__main__':
    app.run(port=3000, debug=True)
