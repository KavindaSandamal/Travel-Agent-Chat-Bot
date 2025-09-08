FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose both Flask (5000) and Streamlit (7000) ports
EXPOSE 5000
EXPOSE 7000

# Run both apps
# Flask app on 0.0.0.0:5000 and Streamlit on 0.0.0.0:7000
CMD python app.py & streamlit run travel_chatbot_app.py --server.address 0.0.0.0 --server.port 7000
