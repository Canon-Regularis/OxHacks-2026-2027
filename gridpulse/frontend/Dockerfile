# syntax=docker/dockerfile:1
# Build from gridpulse/: docker build -f frontend/Dockerfile -t gridpulse-frontend .
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=80
ENV API_PROXY_TARGET=http://backend:8000
ENV PROTOCOL_HEADER=x-forwarded-proto HOST_HEADER=x-forwarded-host
COPY frontend/package*.json ./
RUN npm ci --omit=dev
COPY --from=build /app/build ./build
EXPOSE 80
CMD ["node", "build"]
