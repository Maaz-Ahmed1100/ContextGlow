# we use node 20 here
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

# install node modules
RUN npm install

COPY . .

# start nextjs in dev mode
CMD ["npm", "run", "dev"]
