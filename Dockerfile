# Multistage build for RiskLayer Enterprise
FROM python:3.11-slim as backend
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install .
COPY risklayer /app/risklayer
CMD ["python", "-m", "uvicorn", "risklayer.server.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Separate frontend build
FROM node:20-alpine as frontend-builder
WORKDIR /app
COPY risklayer-dashboard/package.json risklayer-dashboard/package-lock.json* ./
RUN npm install
COPY risklayer-dashboard ./
RUN npm run build

FROM nginx:alpine as frontend
COPY --from=frontend-builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
