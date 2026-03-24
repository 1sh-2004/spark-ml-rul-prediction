FROM python:3.10

# Install Java (fixed package)
RUN apt-get update && apt-get install -y default-jdk

# Set JAVA_HOME automatically
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]