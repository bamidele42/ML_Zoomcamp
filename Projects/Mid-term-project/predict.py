import pickle
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify

dv = DictVectorizer(sparse=False)
with open("trained_model", "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("Payment default")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.fit_transform(customer)

    y_pred = model.predict_proba(X)[0, 1]
    payment_default = y_pred >= 0.5

    result = {
        "Payment_default_probability": float(y_pred),
        "Payment_default": bool(payment_default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
