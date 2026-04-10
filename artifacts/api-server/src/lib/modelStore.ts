import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  LogisticRegressionModel,
  NaiveBayesModel,
  SVMModel,
  computeMetrics,
  trainTestSplit,
  type LogisticRegressionModelState,
  type ModelMetrics,
  type ModelName,
  type NaiveBayesModelState,
  type SVMModelState,
  type SentimentLabel,
} from "./mlModels.js";
import { getDefaultDataset } from "./sampleDataset.js";
import { logger } from "./logger.js";

export interface ModelStore {
  naiveBayes: NaiveBayesModel;
  logisticRegression: LogisticRegressionModel;
  svm: SVMModel;
  metrics: ModelMetrics[];
  bestModel: ModelName;
  lastTrainedAt: Date | null;
  trained: boolean;
}

interface PersistedModelStore {
  version: 1;
  metrics: ModelMetrics[];
  bestModel: ModelName;
  lastTrainedAt: string | null;
  trained: boolean;
  naiveBayes: NaiveBayesModelState;
  logisticRegression: LogisticRegressionModelState;
  svm: SVMModelState;
}

const moduleDir = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(moduleDir, "..", "..");
const dataDir = resolve(packageRoot, ".data");
const modelStorePath = resolve(dataDir, "trained-models.json");

const store: ModelStore = {
  naiveBayes: new NaiveBayesModel(),
  logisticRegression: new LogisticRegressionModel(),
  svm: new SVMModel(),
  metrics: [],
  bestModel: "naive_bayes",
  lastTrainedAt: null,
  trained: false,
};

let initializationPromise: Promise<void> | null = null;

function resetStore(): void {
  store.naiveBayes = new NaiveBayesModel();
  store.logisticRegression = new LogisticRegressionModel();
  store.svm = new SVMModel();
  store.metrics = [];
  store.bestModel = "naive_bayes";
  store.lastTrainedAt = null;
  store.trained = false;
}

function serializeStore(): PersistedModelStore {
  return {
    version: 1,
    metrics: store.metrics,
    bestModel: store.bestModel,
    lastTrainedAt: store.lastTrainedAt?.toISOString() ?? null,
    trained: store.trained,
    naiveBayes: store.naiveBayes.exportState(),
    logisticRegression: store.logisticRegression.exportState(),
    svm: store.svm.exportState(),
  };
}

function hydrateStore(snapshot: PersistedModelStore): void {
  const naiveBayes = new NaiveBayesModel();
  naiveBayes.loadState(snapshot.naiveBayes);

  const logisticRegression = new LogisticRegressionModel();
  logisticRegression.loadState(snapshot.logisticRegression);

  const svm = new SVMModel();
  svm.loadState(snapshot.svm);

  store.naiveBayes = naiveBayes;
  store.logisticRegression = logisticRegression;
  store.svm = svm;
  store.metrics = snapshot.metrics;
  store.bestModel = snapshot.bestModel;
  store.lastTrainedAt = snapshot.lastTrainedAt ? new Date(snapshot.lastTrainedAt) : null;
  store.trained =
    snapshot.trained &&
    store.naiveBayes.isTrained() &&
    store.logisticRegression.isTrained() &&
    store.svm.isTrained();
}

async function persistStore(): Promise<void> {
  if (!store.trained) return;

  await mkdir(dataDir, { recursive: true });
  await writeFile(modelStorePath, JSON.stringify(serializeStore(), null, 2), "utf8");
  logger.info({ modelStorePath }, "Persisted trained models to disk");
}

async function loadPersistedStore(): Promise<void> {
  try {
    const raw = await readFile(modelStorePath, "utf8");
    const snapshot = JSON.parse(raw) as PersistedModelStore;
    hydrateStore(snapshot);
    logger.info(
      {
        modelStorePath,
        bestModel: store.bestModel,
        trained: store.trained,
        metrics: store.metrics.length,
      },
      "Loaded persisted models from disk",
    );
  } catch (error) {
    const maybeError = error as NodeJS.ErrnoException;
    if (maybeError.code === "ENOENT") {
      logger.info({ modelStorePath }, "No persisted models found on disk yet");
      return;
    }

    logger.error({ err: error, modelStorePath }, "Failed to load persisted models");
    resetStore();
  }
}

export async function initializeModelStore(): Promise<void> {
  initializationPromise ??= loadPersistedStore();
  await initializationPromise;
}

export function getModelStore(): ModelStore {
  return store;
}

export async function trainAllModels(texts: string[], labels: SentimentLabel[]): Promise<ModelMetrics[]> {
  const { trainData, trainLabels, testData, testLabels } = trainTestSplit(texts, labels, 0.2);

  logger.info({ trainSize: trainData.length, testSize: testData.length }, "Starting model training");

  const metricsResults: ModelMetrics[] = [];

  const models: Array<{ name: ModelName; model: NaiveBayesModel | LogisticRegressionModel | SVMModel }> = [
    { name: "naive_bayes", model: store.naiveBayes },
    { name: "logistic_regression", model: store.logisticRegression },
    { name: "svm", model: store.svm },
  ];

  for (const { name, model } of models) {
    const startTime = Date.now();
    model.train(trainData, trainLabels);
    const trainingTime = Date.now() - startTime;

    const predictions = testData.map((text) => model.predict(text).label);
    const metrics = computeMetrics(predictions, testLabels, name, trainData.length, trainingTime);
    metricsResults.push(metrics);
    logger.info({ model: name, accuracy: metrics.accuracy }, "Model trained");
  }

  store.metrics = metricsResults;
  store.lastTrainedAt = new Date();
  store.trained = true;
  store.bestModel = metricsResults.reduce((best, metric) => (metric.accuracy > best.accuracy ? metric : best)).model;

  await persistStore();

  return metricsResults;
}

export function getModel(name: ModelName): NaiveBayesModel | LogisticRegressionModel | SVMModel {
  if (name === "naive_bayes") return store.naiveBayes;
  if (name === "logistic_regression") return store.logisticRegression;
  return store.svm;
}

export async function ensureDefaultModelsTrained(): Promise<void> {
  await initializeModelStore();

  if (!store.trained) {
    logger.info("Auto-training models on default dataset");
    const { texts, labels } = getDefaultDataset();
    await trainAllModels(texts, labels);
  }
}
