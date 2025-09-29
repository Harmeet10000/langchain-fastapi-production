

# Build the Docker image (using the project's Dockerfile)
docker build -t langchain-FastAPI-server -f docker/dev.Dockerfile .

# Run the Docker container (detached, port 5000)
docker run -d \
  -p 5000:5000 \
  --name langchain-FastAPI-server \
  --env-file .env.development \
  langchain-FastAPI-server:latest

# View logs
docker logs -f langchain-FastAPI-server

# Tag for Docker Hub
docker tag langchain-FastAPI-server harmeet10000/langchain-FastAPI-server:latest

# Push to Docker Hub
docker push harmeet10000/langchain-FastAPI-server:latest

# (For AWS ECR push replace the tag above with your ECR repo and push)
docker push <your-ecr-repo-uri>

