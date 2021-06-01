import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Getting the data from csv file
def get_data():
    video_data = pd.read_csv('dataset/whc-sites-2019.csv')
    video_data['name_en'] = video_data['name_en'].str.lower()
    return video_data

#Preprocessing the data as per the needs
def combine_data(data):
    data_recommend = data.drop(columns=['region_en', 'unique_number','id_no','rev_bis','name_en','short_description_en',
                                'justification_en','date_inscribed','secondary_dates','danger','date_end','danger_list'
                                ,'longitude','latitude','area_hectares','criteria_txt','category_short','iso_code'
                                ,'udnp_code','transboundary'])


    data_recommend['combine'] = data_recommend[data_recommend.columns[0:3]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

    data_recommend = data_recommend.drop(columns=[ 'category','states_name_en'])

    return data_recommend

#Gettting the cosine cosine_similarity
def transform_data(data_combine, data_plot):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_plot['short_description_en'])

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')

    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

    return cosine_sim

#recommendation data that is to be passed to user
def recommend_videos(title, data, combine, transform):

    indices = pd.Series(data.index, index = data['name_en'])
    index = indices[title]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]

    video_indices = [i[0] for i in sim_scores]
    #print(video_indices)

    video_id = data['unique_number'].iloc[video_indices]
    video_title = data['name_en'].iloc[video_indices]
    video_category = data['category'].iloc[video_indices]
    video_place = data['states_name_en'].iloc[video_indices]

    recommendation_data = pd.DataFrame(columns=['Unique_number','Name','Category','Location'])

    recommendation_data['Unique_number'] = video_id
    recommendation_data['Name'] = video_title
    recommendation_data['Category'] = video_category
    recommendation_data['Location'] = video_place

    return recommendation_data

#Calling function that will be called in this API
def results(video_name):
    video_name = video_name.lower()

    find_video = get_data()
    combine_result = combine_data(find_video)
    transform_result = transform_data(combine_result,find_video)

    if video_name not in find_video['name_en'].unique():
        return 'Video Not Found'

    else:
        recommendations = recommend_videos(video_name, find_video, combine_result, transform_result)
        return recommendations.to_dict('records')
