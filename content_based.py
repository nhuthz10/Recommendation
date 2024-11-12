from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

load_dotenv()

print("DB_USERNAME: " + os.getenv('DB_USERNAME'))
print("DB_PASSWORD: " + os.getenv('DB_PASSWORD'))
print("DB_HOST: " + os.getenv('DB_HOST'))
print("DB_DATABASE" + os.getenv('DB_DATABASE'))

app = Flask(__name__)
CORS(app)  

def get_database_connection():
    db_uri = f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_DATABASE')}"
    engine = create_engine(db_uri)
    return engine

def load_data_from_db():
    engine = get_database_connection()
    query = """
    SELECT T0.productId, T0.productTypeId, T3.productTypeName, T0.brandId, T2.brandName, T0.name, T0.image, T0.price, T0.discount, 
           CASE WHEN AVG(T1.rating) IS NULL THEN 0 ELSE ROUND(AVG(T1.rating), 1) END AS rating, 
           T0.descriptionContent, T0.descriptionHTML 
    FROM products AS T0 
    JOIN brands AS T2 ON T0.brandId = T2.brandId 
    JOIN product_types AS T3 ON T0.productTypeId = T3.productTypeId 
    LEFT JOIN feedbacks AS T1 ON T0.productId = T1.productId 
    GROUP BY T0.productId, T0.productTypeId, T0.brandId, T0.name, T0.image, T0.price, T0.discount, 
             T0.descriptionContent, T0.descriptionHTML, T2.brandName, T3.productTypeName;
    """
    with engine.connect() as connection:
        dataProduct = pd.read_sql(query, connection)
    engine.dispose()
    return dataProduct


def get_content_based_recommendations(product_id, top_n, content_df, content_similarity):
    index = content_df[content_df['productId'] == product_id].index[0]
    similarity_scores = content_similarity[index]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    recommendations = content_df.loc[similar_indices, 'productId'].values
    return recommendations


@app.route('/api/product/recommendation', methods=['GET'])
def get_all_product_recommendation():

    dataProduct = load_data_from_db()

    content_df = dataProduct[['productId', 'name', 'brandName', 'productTypeName']].copy()
    content_df['Content'] = content_df[['name', 'brandName', 'productTypeName']].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    tfidf_vectorizer = TfidfVectorizer()
    content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])
    content_similarity = linear_kernel(content_matrix, content_matrix)

    product_id = request.args.get('productId')
    if not product_id:
        return jsonify({"message": "ProductId is required"}), 400
    if product_id not in content_df['productId'].values:
        return jsonify({"message": "ProductId does not exist"}), 404

    recommendations = get_content_based_recommendations(product_id, 8, content_df, content_similarity)

    recommendations = recommendations.tolist()
    recommendation_info = dataProduct[dataProduct['productId'].isin(recommendations)].to_dict(orient='records')
    data = {"errCode": 0 , "data": recommendation_info}
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=os.getenv('PORT'))
