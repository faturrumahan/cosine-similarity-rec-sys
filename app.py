from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import requests
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

#load data
games_data = pd.read_csv('games.csv',error_bad_lines=False)

games_data = games_data[games_data['Price'] == 0.00] #select only free games (dataset already filtered before)
new_df = games_data[["AppID", "Name", "About the game", "Developers", "Publishers", "Genres", "Tags", "Header image", "Movies"]].copy()
new_df.rename(columns = {'About the game':'About'}, inplace = True)
new_df.rename(columns = {'Header image':'Image'}, inplace = True)

#cleansing
new_df['Name']=new_df['Name'].fillna(0)
new_df['About']=new_df['About'].fillna(0)
new_df['Developers']=new_df['Developers'].fillna(0)
new_df['Publishers']=new_df['Publishers'].fillna(0)
new_df['Genres']=new_df['Genres'].fillna(0)
new_df['Tags']=new_df['Tags'].fillna(0)
new_df['Image']=new_df['Image'].fillna(0)
new_df['Movies']=new_df['Movies'].fillna(0)

@app.route('/', methods=['GET'])
def games():
    result = new_df.to_json(orient="table")
    response = json.loads(result)
    return response
    
@app.route('/search', methods=['GET'])
def search():
    args = request.args

    AppID = int(args.get('AppID')) #retrive id from params
    game_name = new_df[new_df["AppID"] == AppID]["Name"].values[0] #find name of game by id

    selected_features = ['Developers','Publishers','Categories','Genres','Tags'] #select feature for recommendation
    for feature in selected_features:
        games_data[feature] = games_data[feature].fillna('') #replacing null values w/ null string
    combined_features = games_data['Developers']+' '+games_data['Publishers']+' '+games_data['Categories']+' '+games_data['Genres']+' '+games_data['Tags'] #combine all selected feature

    vectorizer = TfidfVectorizer() #vectorize using tfidf vectorizer
    feature_vectors = vectorizer.fit_transform(combined_features) #converting text data to feature vector

    similarity = cosine_similarity(feature_vectors) #get similarity between features

    #giving index to dataframe
    index=[]
    test = 0
    for number in range(games_data.shape[0]):
        index.append(test)
        test = test + 1
    games_data['Index'] = index
    #end of giving index to dataframe

    index_of_the_game = games_data[games_data["Name"] == game_name]['Index'].values[0] #find index of game w/ name

    similarity_score = list(enumerate(similarity[index_of_the_game])) #get list of similarity games
    sorted_similar_games = sorted(similarity_score, key = lambda x:x[1], reverse = True) #sorting the games based on similarity_score

    i = 1
    rec_name = []
    rec_id = []
    rec_image = []

    for game in sorted_similar_games:
        idx = game[0]
        title_from_index = games_data[games_data["Index"]==idx]['Name'].values[0]
        id_from_index = games_data[games_data["Index"]==idx]['AppID'].values[0]
        img_from_index = games_data[games_data["Index"]==idx]['Header image'].values[0]
        if (i<31):
            rec_name.append(title_from_index)
            rec_id.append(id_from_index)
            rec_image.append(img_from_index)
            i+=1

    data = {"AppID": rec_id,
            "Name": rec_name,
            "Image": rec_image}

    rec_df = pd.DataFrame(data)
    result = rec_df.to_json(orient="table")
    response = json.loads(result)
    return response

app.run()