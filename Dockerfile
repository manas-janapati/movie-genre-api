FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*
RUN wget -O movie_genre_model.zip "https://github.com/manas-janapati/movie-genre-api/releases/download/v1.0.0/movie_genre_model.zip" && unzip movie_genre_model.zip && rm movie_genre_model.zip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
