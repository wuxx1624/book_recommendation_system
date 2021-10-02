# book_recommendation_system
A book recommendation system using the goodbook-10k dataset
## Required libraries:
python3,  pandas, numpy, sklearn, pickle, scikit-surprise
## Instruction
1. The folder contains three main items, main.py, utilities.py and model folder
2. 
   The main.py is the primiry script to train or test the system. 
   
   The utilities.py contains some functions required to run the train or test.
   
   The model folder contains pre-trained models.
   
2. Train the model:
3. 
   python main.py train
   
   * there is a pre-trained model in the ./model folder. You can directly test the system.
   
3. Test the model:
   python main.py eval user_id, query_category, query_item(optional)
   
   user_id: the query user id, start from 1
   
   query category can be one of the following: [tag_name, book_id, author, title, popularity, rating]
   
   query item is the one of the item in the query category.
   
   e.g,. query based on tag 'fantasy' for user 1
   
   python main.py eval 1 tag_name fantasy
   
   e.g,. query based on book_id 258 for user 1
   
   python main.py eval 1 book_id 258

   e.g,. query based on popularity for user 1
   
   python main.py eval 1 popularity
   
   * If query based on the popularity or rating, query_item is not needed.
