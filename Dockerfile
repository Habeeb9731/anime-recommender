# Demo image: only the small runtime and the precomputed bundle, no training stack.
FROM python:3.11-slim

WORKDIR /srv/animerec
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY .streamlit .streamlit
COPY app app
COPY src/animerec src/animerec
COPY artifacts/demo artifacts/demo
COPY deploy/brand_static.py deploy/brand_static.py
RUN python deploy/brand_static.py

USER nobody
ENV HOME=/tmp
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
