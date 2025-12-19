from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re

app = Flask(__name__)

# Initialize Spark Session
print("Initializing Spark Session for Web App...")
spark = SparkSession.builder \
    .appName("SentimentWebApp") \
    .master("local[*]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

# Load the trained model
print("Loading trained model...")
model_path = "/app/output/sentiment_model"
try:
    model = PipelineModel.load(model_path)
    print("Model loaded successfully!")
    
    # Get label mapping
    label_mapping = model.stages[4].labels
    print(f"Labels: {label_mapping}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_mapping = None

# Text cleaning function (same as training)
def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_text_udf = udf(clean_text, StringType())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create DataFrame with the input text
        df = spark.createDataFrame([(text,)], ["sentence"])
        
        # Clean the text
        df = df.withColumn("clean_text", clean_text_udf(df.sentence))
        
        # Make prediction
        predictions = model.transform(df)
        
        # Get prediction result
        result = predictions.select("prediction", "probability").collect()[0]
        prediction_idx = int(result.prediction)
        probabilities = result.probability.toArray().tolist()
        
        # Decode label
        predicted_label = label_mapping[prediction_idx]
        confidence = probabilities[prediction_idx] * 100
        
        # Create response with all probabilities
        all_predictions = {
            label_mapping[i]: round(prob * 100, 2) 
            for i, prob in enumerate(probabilities)
        }
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2),
            'all_probabilities': all_predictions,
            'cleaned_text': df.select("clean_text").collect()[0][0]
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'spark_version': spark.version
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)