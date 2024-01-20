from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mysql.connector


logging.basicConfig(level=logging.DEBUG)

DATABASE_URL = "mysql+mysqlconnector://root:0000@localhost:3306/dg"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Post 모델 정의
class Post(Base):
    __tablename__ = "post"
    post_id = Column(Integer, primary_key=True, index=True)
    post_title = Column(String, index=True)


# 데이터베이스 연결 설정
# Base.metadata.create_all(bind=engine)

app = FastAPI()


# DB 세션 의존성 설정
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 모델 파일의 경로를 정확하게 지정
model_path = 'saved_model/my_model'
model = None

try:
    model = tf.keras.models.load_model(model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise HTTPException(status_code=500, detail="Internal Server Error")

# Load the post data
# num_posts = 15
# post_titles = ['전자레인지 판매', '자전거 판매', '책상 판매', '노트북 판매', '의자 판매', '탁자 판매', '램프 판매', '커튼 판매', '침대 판매', '옷장 판매', '신발 판매',
#                '시계 판매', '가방 판매', '키보드 판매', '마우스 판매']
# posts_df = pd.DataFrame({'post_id': range(1, num_posts + 1), 'title': post_titles})


class RecommendationRequest(BaseModel):
    user_id: int
    post_id: int
    interaction: int
    post_title: str
    num_recommendations: int


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list


@app.post("/get_recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    post_id = request.post_id
    interaction = request.interaction
    post_title = request.post_title
    num_recommendations = request.num_recommendations

    logging.info(
        f"Model input: user_id={user_id}, post_id={post_id}, interaction={interaction}, post_title={post_title}, num_recommendations={num_recommendations}")

    try:
        # 게시글 개수 가져오기
        num_posts = db.query(func.count(Post.post_id)).scalar()
        # 게시글 정보 가져오기
        posts = db.query(Post.post_id, Post.post_title).all()
        logging.info(f"num_posts={num_posts}, posts={posts}")
    except Exception as e:
        logging.error(f"DB에서 값 가져오기 실패: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



    # Pandas DataFrame 생성
    posts_df = pd.DataFrame(posts, columns=["post_id", "title"])

    # Generate recommendations using the loaded model
    post_ids = np.array(list(range(1, num_posts + 1)))
    user_ids = np.array([user_id] * num_posts)
    user_ids = np.expand_dims(user_ids, axis=-1)
    post_ids = np.expand_dims(post_ids, axis=-1)
    try:
        predictions = model.predict([user_ids, post_ids]).flatten()
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    top_post_indices = (-predictions).argsort()[:num_recommendations]
    recommended_posts = posts_df.iloc[top_post_indices]

    logging.info(f"Model predictions: {predictions}")

    # Return the recommendations with post_id and post_title
    response = {
        'user_id': user_id,
        'recommendations': [{'post_id': post_id, 'post_title': post_title} for post_id, post_title in
                            zip(recommended_posts['post_id'], recommended_posts['title'])]
    }

    logging.info(f"Model response : {response}")
    return response


# Run the application
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8003)
