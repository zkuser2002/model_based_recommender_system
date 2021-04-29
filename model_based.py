import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
columns=['user_id','item_id','rating','timestamp']
data=pd.read_csv('u.data',sep='\t',names=columns)
columns=['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation','Childrens','Comedy','Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror','Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies=pd.read_csv('u.item',sep='|',names=columns, encoding='latin-1')
movie_names=pd.DataFrame(movies, columns=['item_id','movie title'])
combined_movies_data=pd.merge(data,movies, on='item_id')
rating=pd.DataFrame(combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head())
Filter= combined_movies_data['item_id']==50
print(combined_movies_data[Filter]['movie title'].unique())
rating_crosstab_mat=combined_movies_data.pivot_table(values='rating',index='user_id', columns='movie title',fill_value=0)
X=rating_crosstab_mat.values.T
SVD=TruncatedSVD(n_components=10,random_state=18)
result_mat=SVD.fit_transform(X)
corr_mat=np.corrcoef(result_mat)
movies_names=rating_crosstab_mat.columns
movies_list=list(movies_names)
star_wars= movies_list.index('Star Wars (1977)')
corr_star_wars=corr_mat[star_wars]
print(list(movies_names[(corr_star_wars < 1.0) & (corr_star_wars > 0.9)]))
