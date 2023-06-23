from flask import Flask, render_template, url_for

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/', methods=['GET'])
def get_app():
    css_url = url_for('static', filename='style.css')
    return render_template('index.html', css_url=css_url)


@app.route('/', methods=['POST'])
def move_to_model_1():
    pass


@app.route('/', methods=['POST'])
def move_to_model_2():
    pass


if __name__ == '__main__':
    app.run(port=3000, debug=True)
