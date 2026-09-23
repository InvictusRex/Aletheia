FROM node:22-alpine AS build
WORKDIR /app
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web ./
# NEXT_PUBLIC_* is inlined at build time, so the browser's API origin is
# fixed here. It is the published host port, not the compose hostname:
# the request is made by the browser, not by this container.
ARG NEXT_PUBLIC_ALETHEIA_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_ALETHEIA_API_URL=$NEXT_PUBLIC_ALETHEIA_API_URL
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["npx", "next", "start", "-p", "3000"]
