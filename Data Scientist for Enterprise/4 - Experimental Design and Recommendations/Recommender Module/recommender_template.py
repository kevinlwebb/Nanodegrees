import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
import recommender_functions as rf

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''



    def fit(self, reviews_pth, movies_pth, latent_features = 12, learning_rate = 0.0001, iters = 100):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions

        INPUT
        reviews_pth
        movies_pth
        latent_features
        learning_rate
        iters

        OUTPUT
        None

        SAVED ATTRIBUTES
        n_users
        n_movies
        num_ratings
        reviews
        movies
        user_item_matrix - (np array)
        latent_features - (int)
        learning_rate
        iters
        '''

        self.reviews = pd.read_csv(reviews_pth)
        self.movies = pd.read_csv(movies_pth)

        # Create User Item Matrix
        usr_itm = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = usr_itm.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_matrix = np.array(self.user_item_df)

        # Store More Inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Useful values
        self.n_users = self.user_item_matrix.shape[0]
        self.n_movies = self.user_item_matrix.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_matrix))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # Initialize with random values
        user_mat = np.array(self.n_users, self.latent_features)
        movie_mat = np.array(self.latent_features, self.n_movies)

        # Initiate with 0 for first iteration
        sse_accum = 0

        print("Optimization Statistics")
        print("Iterations | Mean Squared Error")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0
            
            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):
                    
                    # if the rating exists
                    if self.user_item_matrix[i, j] > 0:
                        
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_matrix[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])
                        
                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2
                        
                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))


        # SVD Based Fit
        # keep user matrix and movie matrix
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge Based Fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)




    def predict_rating(self, user_id, movie_id):
        '''
        makes predictions of a rating for a user on a movie-user combo

        INPUT
        user_id
        movie_id

        OUTPUT
        pred
        '''

        try:
            pass
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of row and column in U and V
            pred = np.dot()
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred


        except:
            print("A prediction could not be made for this user movie pair. One of the items is not currently in the database.")

            return None

    def make_recs(self, _id, _id_type="movie", rec_num = 5):
        '''
        given a user id or a movie that an individual likes
        make recommendations

        INPUT
        _id
        _id_type
        rec_num

        OUTPUT
        recs
        '''

        rec_ids, rec_names = None, None

        if _id_type == "user":
            if _id in self.user_ids_series:

                # Get the index of which row the user is in for use in the U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # Take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx, :], self.movie_mat)

                # Get the top movies from the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("This user was not in the database, the system will give top recommendations for all users")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id,self.movies))[:rec_num]
            else:
                print("This movie does not exist in the database. No recommendations can be given")
        
        return rec_ids, rec_names


if __name__ == '__main__':
    # test different parts to make sure it works
