from flask import Flask, request, jsonify
import python_pachyderm
import datetime
from grpc import RpcError


# Pachyderm client to fetch and put file in Pachyderm repos
client = python_pachyderm.PfsClient()
app = Flask(__name__)


def gen_filename(prefix):
    """
    Generate a filename <prefix>-<current time in ISO8601>.txt
    """
    ts = datetime.datetime.now().timestamp()
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    return prefix+"-"+readable+".txt"


@app.route("/train", methods=["POST"])
def add_training_data():
    """
    Submit training sample(s) to `training` repo, which will trigger
    `train` pipeline.

    POST {
        "data": [sample_1, sample_2, ..., sample_n]
    }
    where sample_i: "<user_id> <movie_id> <rate> <timestamp>"
    
    Response: {
        "filename": <created_training_data_filename_in_pachyderm>
    }
    """
    json_data = request.get_json()
    filename = gen_filename("train")
    data = "\n".join(json_data['data']).encode()

    with client.commit('training', 'master') as c:
        client.put_file_bytes(c, filename, data)
    
    return jsonify({"filename": filename})


@app.route("/predict", methods=["POST"])
def predict_data():
    """
    Submit data to predict to `streaming` repo, which will trigger
    `predict` pipeline. Submit is asynchronous and to check the result
    use `POST /result`

    POST {
        "data": [sample_1, sample_2, ..., sample_n]
    } 
    where sample_i: "<user_id> <movie_id>"

    Response: {
        "filename": <created_predicting_data_filename_in_pachyderm>
    }
    """
    json_data = request.get_json()
    filename = gen_filename('predict')
    # pachyderm expect a bytes to save, convert str to bytes
    data = "\n".join(json_data['data']).encode()

    with client.commit('streaming', 'master') as c:
        client.put_file_bytes(c, filename, data)
    
    return jsonify({"filename": filename})


@app.route("/result", methods=["POST"])
def check_result():
    """
    Get the result giving created predicting data filename.

    POST {
        "filename": <filename_returned_by_/predict>
    }

    Response: {
        ready: false
    } or {
        ready: true
        data: <content-of-result-file>
    }
    """
    json_data = request.get_json()
    filename = json_data['filename']
    try:
        data = client.get_file("predict/master", filename)
        return jsonify({
            "ready": True,
            # Data is a iterator of bytes object, convert to str
            "data": list(data)[0].decode()
        })
    except RpcError:
        # If file not exist, either it's a wrong filename or 
        # predict-pipeline didn't finish processing
        return jsonify({"ready": False})
