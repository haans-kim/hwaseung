# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files from frontend directory
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend app files
COPY frontend/ .

# Copy database file from project root (resolve symlink issue)
COPY hwaseung_RnD.db ./public/hwaseung_RnD.db

# Build the app (API URL will be dynamically determined at runtime)
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app
COPY --from=build /app/build /usr/share/nginx/html

# Add nginx config (from frontend directory since context is project root)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]