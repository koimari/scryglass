# Scryglass

A small League of Legends rankings site.

## Public pages

- `/` redirects to team and player ratings.
- `/elo` shows team and player ratings.
- `/tiers` shows champion tier lists.
- `/methodology` explains both ranking methods.

## Local use

```bash
cd apps/scryglass
npm install
npm run dev
```

Ratings use a versioned JSON snapshot. The server checks for a new accepted
snapshot every six hours. The deployed copy is the fallback during a storage
outage.

The public app excludes raw game rows, training data, coefficients, research
studies, prediction artifacts, and betting tools.
