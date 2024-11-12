FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3010
ENV FLASK_APP=content_based.py
ENV FLASK_ENV=production

CMD ["flask", "run", "--host=0.0.0.0", "--port=3010"]
