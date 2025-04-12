FROM diffusers/diffusers-pytorch-cuda


WORKDIR /app

# Copy only the uv.lock first for caching dependencies

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the rest of the codebase
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
