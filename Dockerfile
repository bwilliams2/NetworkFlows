FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /

RUN pip install -r /requirements.txt \
	&& rm -rf /root/.cache

COPY ./ ./

EXPOSE 8085

WORKDIR /app/analysis
CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
 