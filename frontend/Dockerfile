FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
# NEXT_PUBLIC_* is inlined at build time, so the browser's API origin is
# fixed here. It is the published host port, not the compose hostname:
# the request is made by the browser, not by this container.
ARG NEXT_PUBLIC_ALETHEIA_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_ALETHEIA_API_URL=$NEXT_PUBLIC_ALETHEIA_API_URL
# Where the /api rewrite forwards to. Resolved into the build output, so
# it is an argument here and not a runtime variable.
ARG ALETHEIA_API_URL=http://localhost:8000
ENV ALETHEIA_API_URL=$ALETHEIA_API_URL
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
