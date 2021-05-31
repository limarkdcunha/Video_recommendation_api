import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data():
    movie_data = pd.read_csv('dataset/whc-sites-2019.csv')
    movie_data['name_en'] = movie_data['name_en'].str.lower()
    return movie_data


def combine_data(data):
    data_recommend = data.drop(columns=['region_en', 'unique_number','id_no','rev_bis','name_en','short_description_en',
                                'justification_en','date_inscribed','secondary_dates','danger','date_end','danger_list'
                                ,'longitude','latitude','area_hectares','criteria_txt','category_short','iso_code'
                                ,'udnp_code','transboundary'])


    data_recommend['combine'] = data_recommend[data_recommend.columns[0:3]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

    #print('Hola\n\n\n\n')
    #print(data_recommend)

    data_recommend = data_recommend.drop(columns=[ 'category','states_name_en'])

    #print('\n\n\nHola2\n\n\n\n')
    #print(data_recommend)
    return data_recommend


def transform_data(data_combine, data_plot):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_plot['short_description_en'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')

    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim


def recommend_movies(title, data, combine, transform):

    indices = pd.Series(data.index, index = data['name_en'])
    index = indices[title]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]

    movie_indices = [i[0] for i in sim_scores]

    movie_id = data['unique_number'].iloc[movie_indices]
    movie_title = data['name_en'].iloc[movie_indices]
    movie_genres = data['category'].iloc[movie_indices]

    recommendation_data = pd.DataFrame(columns=['Unique_number','Name','Category'])

    recommendation_data['Unique_number'] = movie_id
    recommendation_data['Name'] = movie_title
    recommendation_data['Category'] = movie_genres

    return recommendation_data

def results(movie_name):
    movie_name = movie_name.lower()

    find_movie = get_data()
    combine_result = combine_data(find_movie)
    transform_result = transform_data(combine_result,find_movie)

    if movie_name not in find_movie['name_en'].unique():
        return 'Movie not in Database'

    else:
        recommendations = recommend_movies(movie_name, find_movie, combine_result, transform_result)
        return recommendations.to_dict('records')
