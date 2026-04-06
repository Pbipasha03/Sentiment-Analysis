# Workspace

## Overview

Microtext Sentiment Analysis — a full-stack web application for NLP-based sentiment analysis of tweets and custom text. Built as a final year academic project using traditional ML models (Naive Bayes, Logistic Regression, SVM) implemented in pure TypeScript on the backend.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Frontend**: React + Vite + Tailwind CSS + Recharts

## Architecture

### Frontend (`artifacts/sentiment-analysis/`)
- React + Vite web app at `/` (root)
- 7 pages: Dashboard, Analyze Text, Batch Analysis, Model Performance, Compare Models, Word Cloud, Export Report
- Recharts for all data visualizations (bar charts, pie charts, confusion matrix heatmap)
- Custom word cloud using flex-wrap with font sizes proportional to frequency

### Backend (`artifacts/api-server/`)
Core ML library is pure TypeScript, no Python required:
- `src/lib/nlpUtils.ts` — Text preprocessing, tokenization, TF-IDF, stopwords, bigrams
- `src/lib/mlModels.ts` — Naive Bayes, Logistic Regression, SVM implementations
- `src/lib/modelStore.ts` — In-memory model state, training orchestration
- `src/lib/sampleDataset.ts` — 100 pre-labeled Twitter-style sample texts
- `src/routes/sentiment.ts` — All 8 API endpoints

### Database (`lib/db/`)
- `sentimentAnalysis.ts` — Tables for analysis results and model training sessions

## Key Features
- Single text analysis with all 3 models simultaneously
- Batch analysis from CSV upload or pre-loaded sample dataset
- Model training with 80/20 train-test split, auto-trains on first use
- Confusion matrix, accuracy, precision, recall, F1 score per model
- Side-by-side model comparison with consensus voting
- Word cloud with sentiment filter
- Downloadable reports in CSV and JSON

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
