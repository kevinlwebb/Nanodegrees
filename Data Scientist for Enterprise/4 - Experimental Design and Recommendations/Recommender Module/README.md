# Recommender Module

## Instructions
Open python in the terminal
```
python
```

Type the following
```
from recommender import Recommender

rec = Recommender()

rec.fit('train_data.csv', 'movies_clean.csv')

...

rec.n_users

rec.n_movies

rec.make_recommendations(37287, _id_type='user')
```
