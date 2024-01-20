import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dot, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

num_users = 10
num_posts = 15
embedding_size = 5

np.random.seed(42)
user_ids = np.random.randint(1, num_users + 1, size=100)
post_ids = np.random.randint(1, num_posts + 1, size=100)
interactions = np.random.randint(0, 2, size=100)
interactions_df = pd.DataFrame({'user_id': user_ids, 'post_id': post_ids, 'interaction': interactions})

post_titles = ['전자레인지 판매', '자전거 판매', '책상 판매', '노트북 판매', '의자 판매', '탁자 판매', '램프 판매', '커튼 판매', '침대 판매', '옷장 판매', '신발 판매',
               '시계 판매', '가방 판매', '키보드 판매', '마우스 판매']
posts_df = pd.DataFrame({'post_id': range(1, num_posts + 1), 'title': post_titles})

user_input = Input(shape=(1,), name='user_input')
post_input = Input(shape=(1,), name='post_input')

user_embedding = Embedding(num_users + 1, embedding_size, input_length=1, name='user_embedding',
                           embeddings_regularizer=l2(0.001))(user_input)
post_embedding = Embedding(num_posts + 1, embedding_size, input_length=1, name='post_embedding',
                           embeddings_regularizer=l2(0.001))(post_input)

flattened_user_embedding = Flatten()(user_embedding)
flattened_post_embedding = Flatten()(post_embedding)

merged_vectors = Dot(axes=1)([flattened_user_embedding, flattened_post_embedding])
flattened_vectors = Flatten()(merged_vectors)
dropout_layer = Dropout(0.5)(flattened_vectors)
output = Dense(1, activation='sigmoid')(dropout_layer)
model = Model(inputs=[user_input, post_input], outputs=output)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(interactions_df[['user_id', 'post_id']],
                                                    interactions_df['interaction'], test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit([X_train['user_id'], X_train['post_id']], y_train, epochs=100, batch_size=10, validation_split=0.2,
          callbacks=[early_stopping])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

y_pred = model.predict([X_test['user_id'], X_test['post_id']])
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error on Test Set:', mse)


def get_recommendations(user_id, model, posts_df, num_recommendations=5):
    post_ids = np.array(list(range(1, num_posts + 1)))
    user_ids = np.array([user_id] * num_posts)
    user_ids = np.expand_dims(user_ids, axis=-1)
    post_ids = np.expand_dims(post_ids, axis=-1)
    predictions = model.predict([user_ids, post_ids]).flatten()
    top_post_indices = (-predictions).argsort()[:num_recommendations]
    recommended_posts = posts_df.iloc[top_post_indices]
    return recommended_posts['title']


user_id_to_recommend = 1
recommended_titles = get_recommendations(user_id_to_recommend, model, posts_df)
print('Recommended Posts for User {}:'.format(user_id_to_recommend))
print(recommended_titles)

model.save('saved_model/my_model', save_format='tf')
