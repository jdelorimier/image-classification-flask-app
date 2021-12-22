FROM python:3.8-slim-buster
LABEL maintainer="delorimier16@gmail.com"
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt
COPY image_classification_flask_app image_classification_flask_app/ 
RUN python image_classification_flask_app/main.py
EXPOSE 8080
CMD ["python", "image_classification_flask_app/main.py", "serve"]