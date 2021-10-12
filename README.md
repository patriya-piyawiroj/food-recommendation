# food-recommendation
Combined collaborative filtering and word embeddings to create a mobile app recommending food based on the food you've previously liked!

See docs for UMLs, app screenshots, and final project presentation

```
app: mobile application portion
food2vec: recommendation model
```

# App
Swipe left and right to build a taste preference. Discover new restaurants based on recommendations

# food2vec
Approaches:
Collaborative Filtering:  Rely on historical usage data to make predictions: similar users likely enjoy same foods
Drawback: Requires enormous dataset of user preferences: does not publicly exist in a structured format
Content-Based System: Make predictions based on the properties of each food item (ingredients): identify similarity between foods
Advantage: Large dataset of 1 million recipes with ingredient information is available; can learn the semantics of recipes

Methods:
Trained word2vec model on the cleaned recipe dataset (> 1 million recipes) using the Gensim library in Python
Embedding size: 100 (size of the embedding vector for each ingredient)
Context-size: 10 words (size of context window around each word)
10 iterations/epochs (number of passes over the entire training set)
Entire training process finished in about 5 minutes on my CPU
Output food2vec model is only 4 MB in size (lightning-quick prediction time)

Results:
Query: dried rice noodles
Most similar food concepts: dried rice, garlic chili sauce, asian, nuoc mam, mung bean
Query: granola
Most similar food concepts: oatmeal, cereal flakes, fiber, dried blueberries, banana 
