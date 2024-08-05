from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import pickle

app = Flask(__name__)

# Load the datasets
drugbank_df = pd.read_csv('drugbank vocabulary.csv')
ddinter_df = pd.read_csv('ddinter_downloads_code_A.csv')
ddinter_df1 = pd.read_csv('ddinter_downloads_code_B.csv')
ddinter_df2 = pd.read_csv('ddinter_downloads_code_V.csv')
ddinter_df = pd.concat([ddinter_df, ddinter_df1, ddinter_df2], ignore_index=True)

# Preprocess DrugBank vocabulary dataset
drugbank_df.columns = ['DrugBank_ID', 'Accession_Numbers', 'Common_name', 'CAS', 'UNII', 'Synonyms', 'Standard_InChI_Key']
drugbank_df = drugbank_df[['Common_name', 'Synonyms']]
drugbank_df['Synonyms'] = drugbank_df['Synonyms'].fillna('')

# Function to resolve synonyms
def resolve_synonyms(df):
    drug_synonym_dict = {}
    for _, row in df.iterrows():
        common_name = row['Common_name']
        synonyms = str(row['Synonyms']).split(' | ')
        for synonym in synonyms:
            drug_synonym_dict[synonym.strip()] = common_name
        drug_synonym_dict[common_name] = common_name
    return drug_synonym_dict

# Create a synonym resolution dictionary
synonym_dict = resolve_synonyms(drugbank_df)

# Map synonyms in DDInter dataset
def map_synonyms(df1, synonym_dict):
    df1['Drug_A'] = df1['Drug_A'].map(synonym_dict).fillna(df1['Drug_A'])
    df1['Drug_B'] = df1['Drug_B'].map(synonym_dict).fillna(df1['Drug_B'])
    return df1

data = map_synonyms(ddinter_df, synonym_dict)

# Load encoders and model
# Load the encoders
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open('onehot_encoder.pkl', 'rb') as ohe_file:
    onehot_encoder = pickle.load(ohe_file)

model = load_model('drug_interaction_model.h5')

# model = load_model('drug_interaction_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Invalid input'}), 400

    drug_a = data.get('drug_a')
    drug_b = data.get('drug_b')

    if not drug_a or not drug_b:
        return jsonify({'error': 'Missing Drug_A or Drug_B'}), 400

    def get_prediction(drug_a, drug_b):
        input_data = pd.DataFrame({'Drug_A': [drug_a], 'Drug_B': [drug_b]})
        
        # Handle possible column names
        input_data = input_data.rename(columns=lambda x: x.strip())

        for column in input_data.columns:
            input_data[column] = input_data[column].apply(lambda x: synonym_dict.get(x, x))
        
        # Ensure the columns are in the same order as the model expects
        try:
            input_data_encoded = onehot_encoder.transform(input_data)
        except Exception as e:
            return {'error': f'Encoding error: {str(e)}'}
        
        try:
            prediction = model.predict(input_data_encoded)
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}

        # Decode the prediction
        try:
            prediction_label = label_encoder.inverse_transform([np.argmax(prediction)])
        except Exception as e:
            return {'error': f'Decoding error: {str(e)}'}
        
        return prediction_label[0]

    prediction1 = get_prediction(drug_a, drug_b)
    prediction2 = get_prediction(drug_b, drug_a)

    if 'error' in prediction1 and 'error' in prediction2:
        return jsonify({'error': 'Both predictions failed'}), 500

    response = {
        'prediction_ab': prediction1 if prediction1 != 'Unknown' else 'Unknown interaction',
        'prediction_ba': prediction2 if prediction2 != 'Unknown' else 'Unknown interaction'
    }
    final_response = {}

    for key in response:
        if response[key] != 'Unknown interaction':
            final_response[key] = response[key]
            break
    
    response = final_response if final_response else {'prediction': 'Unknown'}
            

    return jsonify(response)

# Function to get neighbors and their interaction levels
def get_neighbors(drug):
    neighbors = []
    for _, row in ddinter_df.iterrows():
        if row['Drug_A'] == drug:
            neighbors.append((row['Drug_B'], row['Level']))
        elif row['Drug_B'] == drug:
            neighbors.append((row['Drug_A'], row['Level']))
    return neighbors

@app.route('/graph', methods=['POST'])
def graph():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Invalid input'}), 400

    drug_a = data.get('Drug_A')
    drug_b = data.get('Drug_B')
    interaction_level = data.get('Level')

    if not drug_a or not drug_b or not interaction_level:
        return jsonify({'error': 'Missing Drug_A, Drug_B, or Level'}), 400

    level_mapping = {
        'Minor': 1,
        'Moderate': 2,
        'Major': 3
    }

    weight = level_mapping.get(interaction_level, 1)  # Default to 1 if the level is not found

    G = nx.Graph()
    G.add_edge(drug_a, drug_b, weight=weight)

    # Add neighbors and their interactions
    for drug in [drug_a, drug_b]:
        neighbors = get_neighbors(drug)
        for neighbor, level in neighbors:
            neighbor_weight = level_mapping.get(level, 1)
            G.add_edge(drug, neighbor, weight=neighbor_weight)

    plt.figure(figsize=(10, 10))  # Increase the figure size
    pos = nx.spring_layout(G)
    
    # Define edge colors based on weight
    edge_colors = ['green' if weight == 1 else 'orange' if weight == 2 else 'red' for _, _, weight in G.edges(data='weight')]
    
    nx.draw(G, pos, with_labels=True, node_size=300, node_color="blue", font_size=10, font_color="black", edge_color=edge_colors, width=2)  # Decrease node size

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return f"<img src='data:image/png;base64,{img_base64}'>"

if __name__ == '__main__':
    app.run(debug=True)