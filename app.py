from flask import Flask

from src.business_logic.process_query import create_business_logic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Please use other route to search for desired ticker!:\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction = bl.do_predictions_for(ticker)
    test = "test"

    return f'{prediction}\n'

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
