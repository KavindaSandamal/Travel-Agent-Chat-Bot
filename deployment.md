# AWS EC2 Deployment Guide for Travel Agent Chat Bot

## 1. Build Docker Image Locally

```bash
docker build -t travel-agent-chatbot .
```

## 2. Test Locally (Optional)

```bash
docker run -p 5000:5000 -p 7000:7000 travel-agent-chatbot
```

Visit:

* Flask API: `http://localhost:5000`
* Streamlit UI: `http://localhost:7000`

## 3. Launch an EC2 Instance

* Go to AWS Console > EC2 > Launch Instance.
* Choose Ubuntu Server (recommended).
* Select instance type (e.g., t2.micro for testing).
* Configure security group: Add inbound rules for **TCP ports 5000 and 7000**.

## 4. Connect to EC2

```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

## 5. Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```

Log out and back in to apply Docker group permissions.

## 6. Transfer Project to EC2

From your local machine:

```bash
scp -i /path/to/your-key.pem -r /path/to/Travel-Agent-Chat-Bot ubuntu@<EC2-PUBLIC-IP>:~/Travel-Agent-Chat-Bot
```

## 7. Build and Run Docker Container on EC2

```bash
cd ~/Travel-Agent-Chat-Bot
docker build -t travel-agent-chatbot .
docker run -d -p 5000:5000 -p 7000:7000 --name travel-agent-chatbot travel-agent-chatbot
```

## 8. Access Your Web App

Visit:

* Flask API: `http://<EC2-PUBLIC-IP>:5000`
* Streamlit UI: `http://<EC2-PUBLIC-IP>:7000`

---

For production, consider setting up **Nginx as a reverse proxy** and enabling **HTTPS**.
