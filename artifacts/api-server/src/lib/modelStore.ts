import { NaiveBayesModel, LogisticRegressionModel, SVMModel, computeMetrics, trainTestSplit } from "./mlModels.js";
import type { SentimentLabel, ModelName, ModelMetrics } from "./mlModels.js";
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

const store: ModelStore = {
  naiveBayes: new NaiveBayesModel(),
  logisticRegression: new LogisticRegressionModel(),
  svm: new SVMModel(),
  metrics: [],
  bestModel: "naive_bayes",
  lastTrainedAt: null,
  trained: false,
};

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

    const predictions = testData.map(t => model.predict(t).label);
    const metrics = computeMetrics(predictions, testLabels, name, trainData.length, trainingTime);
    metricsResults.push(metrics);
    logger.info({ model: name, accuracy: metrics.accuracy }, "Model trained");
  }

  store.metrics = metricsResults;
  store.lastTrainedAt = new Date();
  store.trained = true;
  store.bestModel = metricsResults.reduce((best, m) => (m.accuracy > best.accuracy ? m : best)).model;

  return metricsResults;
}

export function getModel(name: ModelName): NaiveBayesModel | LogisticRegressionModel | SVMModel {
  if (name === "naive_bayes") return store.naiveBayes;
  if (name === "logistic_regression") return store.logisticRegression;
  return store.svm;
}

export async function ensureDefaultModelsTrained(): Promise<void> {
  if (!store.trained) {
    logger.info("Auto-training models on default dataset");
    const { texts, labels } = getDefaultDataset();
    await trainAllModels(texts, labels);
  }
}
