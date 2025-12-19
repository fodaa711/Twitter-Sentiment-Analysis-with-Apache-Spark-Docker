\# 🐦 Twitter Sentiment Analysis with Apache Spark \& Docker



A distributed sentiment analysis system using Apache Spark MLlib for classifying Twitter data into four categories: \*\*Positive\*\*, \*\*Negative\*\*, \*\*Neutral\*\*, and \*\*Irrelevant\*\*. Includes a web interface for real-time predictions.



!\[Python](https://img.shields.io/badge/Python-3.11-blue)

!\[Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange)

!\[Docker](https://img.shields.io/badge/Docker-Latest-blue)

!\[Flask](https://img.shields.io/badge/Flask-3.0.0-green)



---



\## 📊 \*\*Project Overview\*\*



This project implements a complete end-to-end machine learning pipeline for sentiment analysis:



\- \*\*Distributed Processing\*\*: Apache Spark cluster with 1 master and 2 workers

\- \*\*Machine Learning\*\*: Logistic Regression with TF-IDF feature extraction

\- \*\*Containerization\*\*: Fully Dockerized application

\- \*\*Web Interface\*\*: Flask-based UI for real-time predictions

\- \*\*High Accuracy\*\*: 91.74% training accuracy, 80.18% validation accuracy



---



\## 🏗️ \*\*Architecture\*\*



```

┌─────────────────────────────────────────────────────────┐

│                   Docker Containers                     │

├─────────────────────────────────────────────────────────┤

│                                                         │

│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │

│  │ Spark Master │  │ Spark Worker │  │ Spark Worker │ │

│  │   (port 8080)│  │      1       │  │      2       │ │

│  └──────────────┘  └──────────────┘  └──────────────┘ │

│         │                  │                  │        │

│         └──────────────────┴──────────────────┘        │

│                         │                              │

│                  ┌──────────────┐                      │

│                  │  Web App     │                      │

│                  │ (port 5000)  │                      │

│                  └──────────────┘                      │

│                         │                              │

└─────────────────────────┼───────────────────────────────┘

&nbsp;                         │

&nbsp;                  ┌──────┴──────┐

&nbsp;                  │   Browser   │

&nbsp;                  └─────────────┘

```



---



\## 🚀 \*\*Quick Start\*\*



\### \*\*Prerequisites\*\*



\- Docker Desktop installed and running

\- Git installed

\- 4GB+ RAM available



\### \*\*1. Clone the Repository\*\*



```bash

git clone https://github.com/fodaa711/twitter-sentiment-spark.git

cd twitter-sentiment-spark

```



\### \*\*2. Download Dataset\*\*



Download the Twitter Entity Sentiment Analysis dataset from \[Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) and place the CSV files in the `data/` folder:



\- `twitter\_training.csv`

\- `twitter\_validation.csv`



\### \*\*3. Build Docker Images\*\*



```bash

docker-compose build

```



This takes 3-5 minutes on first build.



\### \*\*4. Start the Application\*\*



```bash

docker-compose up -d

```



\### \*\*5. Train the Model\*\*



```bash

docker exec -it spark-sentiment-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /app/src/sentiment\_analysis.py

```



Training takes approximately 5 minutes.



\### \*\*6. Access the Web Interface\*\*



Open your browser:

```

http://localhost:5000

```



---



\## 📁 \*\*Project Structure\*\*



```

twitter-sentiment-spark/

├── Dockerfile                  # Spark cluster container

├── Dockerfile.webapp           # Web app container

├── docker-compose.yml          # Docker orchestration

├── requirements.txt            # Python dependencies for Spark

├── webapp-requirements.txt     # Python dependencies for web app

├── .gitignore                  # Git ignore rules

├── README.md                   # This file

│

├── src/                        # Source code

│   └── sentiment\_analysis.py  # Main Spark ML pipeline

│

├── webapp/                     # Web application

│   ├── app.py                 # Flask backend

│   ├── templates/

│   │   └── index.html         # Web interface

│   └── static/

│       └── style.css          # Styling

│

├── data/                       # Dataset (not tracked in Git)

│   ├── twitter\_training.csv

│   └── twitter\_validation.csv

│

├── output/                     # Generated files (not tracked in Git)

│   ├── sentiment\_model/       # Trained Spark model

│   └── validation\_predictions/ # Prediction results

│

└── models/                     # Pre-trained models (optional)

```



---



\## 🛠️ \*\*ML Pipeline Details\*\*



\### \*\*Data Processing\*\*

1\. \*\*Text Cleaning\*\*: Remove URLs, special characters, lowercase

2\. \*\*Tokenization\*\*: Split text into words

3\. \*\*Stop Words Removal\*\*: Remove common English words

4\. \*\*TF-IDF Vectorization\*\*: Convert text to numerical features (20,000 features)



\### \*\*Model\*\*

\- \*\*Algorithm\*\*: Logistic Regression

\- \*\*Max Iterations\*\*: 20

\- \*\*Regularization\*\*: 0.01

\- \*\*Classes\*\*: 4 (Positive, Negative, Neutral, Irrelevant)



\### \*\*Performance Metrics\*\*

\- \*\*Training Accuracy\*\*: 91.74%

\- \*\*Validation Accuracy\*\*: 80.18%

\- \*\*Training Samples\*\*: 74,682 tweets

\- \*\*Validation Samples\*\*: 1,510 tweets



---



\## 🌐 \*\*Web Interface Features\*\*



\- \*\*Real-time Predictions\*\*: Enter text and get instant sentiment analysis

\- \*\*Confidence Scores\*\*: See probability distribution across all classes

\- \*\*Example Buttons\*\*: Quick test with pre-defined examples

\- \*\*Clean Text Display\*\*: View how text is processed before prediction

\- \*\*Responsive Design\*\*: Works on desktop and mobile devices



---



\## 📊 \*\*Monitoring\*\*



\### \*\*Spark Master UI\*\*

```

http://localhost:8080

```

View cluster status, workers, and active jobs.



\### \*\*Spark Application UI\*\* (during job execution)

```

http://localhost:4040

```

Monitor stages, tasks, and execution details.



---



\## 🐳 \*\*Docker Commands\*\*



\### \*\*Start Application\*\*

```bash

docker-compose up -d

```



\### \*\*Stop Application\*\*

```bash

docker-compose down

```



\### \*\*View Logs\*\*

```bash

docker-compose logs -f

```



\### \*\*Restart Containers\*\*

```bash

docker-compose restart

```



\### \*\*Rebuild After Code Changes\*\*

```bash

docker-compose down

docker-compose build

docker-compose up -d

```



---



\## 🔧 \*\*Development\*\*



\### \*\*Updating Code\*\*



Since `src/` and `webapp/` folders are mounted as volumes, you can:

1\. Edit code on your computer

2\. Restart containers: `docker-compose restart`

3\. Changes take effect immediately (no rebuild needed)



\### \*\*Adding New Dependencies\*\*



If you add new Python packages:

1\. Update `requirements.txt` or `webapp-requirements.txt`

2\. Rebuild: `docker-compose build`

3\. Restart: `docker-compose up -d`



---



\## 📈 \*\*Results\*\*



The model successfully classifies tweets into four sentiment categories with high accuracy:



| Metric | Score |

|--------|-------|

| Training Accuracy | 91.74% |

| Validation Accuracy | 80.18% |

| Model | Logistic Regression |

| Features | 20,000 TF-IDF |



\### \*\*Example Predictions\*\*



| Text | Predicted Sentiment |

|------|---------------------|

| "This product is amazing!" | Positive |

| "Terrible experience, waste of money" | Negative |

| "The weather is nice today" | Neutral |

| "Random discussion about nothing" | Irrelevant |



---



\## 🧪 \*\*Testing\*\*



Test the API endpoint directly:



```bash

curl -X POST http://localhost:5000/predict \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"text":"I love this product!"}'

```



---



\## 🤝 \*\*Contributing\*\*



Contributions are welcome! Please feel free to submit a Pull Request.



---



\## 📄 \*\*License\*\*



This project is for educational purposes.



---



\## 👨‍💻 \*\*Author\*\*



\*\*Mohamed Fode\*\*



Big Data Project - Sentiment Analysis with Apache Spark



---



\## 🙏 \*\*Acknowledgments\*\*



\- Dataset: \[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) from Kaggle

\- Apache Spark MLlib for distributed machine learning

\- Docker for containerization

\- Flask for web framework



---



\## 📞 \*\*Support\*\*



For issues or questions, please open an issue in the GitHub repository.



---



\*\*Built with ❤️ using Apache Spark, Docker, and Flask\*\*

