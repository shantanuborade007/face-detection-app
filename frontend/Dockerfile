FROM node:18

# Set working directory
WORKDIR /app

# Install dependencies early to leverage Docker cache
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Start Vite dev server
CMD ["npm", "run", "dev", "--", "--host"]
