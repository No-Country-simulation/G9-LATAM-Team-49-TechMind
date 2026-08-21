FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci || npm install
COPY . .
# Vacio => el frontend llama a rutas relativas y nginx hace de proxy hacia la
# API. Solo se define si se quiere apuntar a un host distinto.
ARG PUBLIC_API_URL=""
ENV PUBLIC_API_URL=$PUBLIC_API_URL
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
