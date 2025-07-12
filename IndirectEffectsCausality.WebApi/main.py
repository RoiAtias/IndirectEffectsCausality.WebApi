from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from IndirectEffectsLogic import IndirectEffectsLogic

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})
indirect_effects_logic = IndirectEffectsLogic();


@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return "i'm alive!";

@app.route('/api/indirectEffects/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            data = indirect_effects_logic.process_csv_file(file)
            return jsonify({'message': 'File successfully processed', 'data': data}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return jsonify({'error': 'Only CSV files are allowed'}), 400

@app.route('/api/indirectEffects/results', methods=['POST'])
def send_results():
    file = request.files.get('file')
    confounders = request.form.get('confounders')
    predictor_x = request.form.get('predictorX')
    mediator_y = request.form.get('mediatorY')
    target_variable = request.form.get('targetVariable')
    mediator_model = request.form.get('mediatorModel')
    target_model = request.form.get('targetModel')

    try:
        result = indirect_effects_logic.process_results(file, confounders, predictor_x, mediator_y, target_variable,mediator_model,target_model)
        return jsonify({'message': 'Results processed successfully', 'data': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'success', 'message': 'Test endpoint is working'})

if __name__ == '__main__':
    app.run(debug=False)