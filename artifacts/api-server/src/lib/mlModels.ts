import {
  preprocessText,
  tokenize,
  buildVocabulary,
  computeIdf,
  textToTfIdf,
  computeBagOfWords,
  createBigrams,
} from "./nlpUtils.js";

export type SentimentLabel = "positive" | "negative" | "neutral";
export type ModelName = "naive_bayes" | "logistic_regression" | "svm";

export interface SentimentResult {
  label: SentimentLabel;
  confidence: number;
  scores: { positive: number; negative: number; neutral: number };
  model: ModelName;
  processedText: string;
}

export interface ModelMetrics {
  model: ModelName;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: { matrix: number[][]; labels: string[] };
  classReport: Record<string, { precision: number; recall: number; f1Score: number; support: number }>;
  trainingSamples: number;
  testSamples: number;
  trainingTimeMs: number;
}

const LABELS: SentimentLabel[] = ["positive", "negative", "neutral"];
const LABEL_INDEX: Record<SentimentLabel, number> = { positive: 0, negative: 1, neutral: 2 };

function softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0);
}

function addVectors(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + (b[i] || 0));
}

function scaleVector(v: number[], s: number): number[] {
  return v.map(x => x * s);
}

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map(x => x / norm);
}

export class NaiveBayesModel {
  private logPriors: Record<SentimentLabel, number> = { positive: 0, negative: 0, neutral: 0 };
  private logLikelihoods: Record<SentimentLabel, number[]> = {
    positive: [], negative: [], neutral: [],
  };
  private vocab: string[] = [];
  private trained = false;

  train(texts: string[], labels: SentimentLabel[]): void {
    this.vocab = buildVocabulary(texts);
    const V = this.vocab.length;

    const classCounts: Record<SentimentLabel, number> = { positive: 0, negative: 0, neutral: 0 };
    const wordCounts: Record<SentimentLabel, number[]> = {
      positive: new Array(V).fill(0),
      negative: new Array(V).fill(0),
      neutral: new Array(V).fill(0),
    };

    for (let i = 0; i < texts.length; i++) {
      const label = labels[i];
      classCounts[label]++;
      const bow = computeBagOfWords(texts[i], this.vocab);
      for (let j = 0; j < V; j++) {
        wordCounts[label][j] += bow[j];
      }
    }

    const N = texts.length;
    for (const label of LABELS) {
      this.logPriors[label] = Math.log((classCounts[label] + 1) / (N + LABELS.length));
      const totalWords = wordCounts[label].reduce((s, c) => s + c, 0);
      this.logLikelihoods[label] = wordCounts[label].map(c =>
        Math.log((c + 1) / (totalWords + V))
      );
    }
    this.trained = true;
  }

  predict(text: string): SentimentResult {
    if (!this.trained) throw new Error("Model not trained");
    const bow = computeBagOfWords(text, this.vocab);
    const rawScores = LABELS.map(label => {
      let score = this.logPriors[label];
      for (let j = 0; j < this.vocab.length; j++) {
        if (bow[j] > 0) {
          score += this.logLikelihoods[label][j] * bow[j];
        }
      }
      return score;
    });

    const probs = softmax(rawScores);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: LABELS[maxIdx],
      confidence: Math.round(probs[maxIdx] * 1000) / 1000,
      scores: { positive: probs[0], negative: probs[1], neutral: probs[2] },
      model: "naive_bayes",
      processedText: preprocessText(text),
    };
  }

  isTrained(): boolean { return this.trained; }
}

export class LogisticRegressionModel {
  private weights: number[][] = [];
  private biases: number[] = [];
  private vocab: string[] = [];
  private idf: Record<string, number> = {};
  private trained = false;

  train(texts: string[], labels: SentimentLabel[]): void {
    this.vocab = buildVocabulary(texts);
    this.idf = computeIdf(texts, this.vocab);
    const V = this.vocab.length;
    const K = LABELS.length;
    const lr = 0.1;
    const epochs = 200;
    const lambda = 0.001;

    this.weights = Array.from({ length: K }, () => new Array(V).fill(0).map(() => (Math.random() - 0.5) * 0.01));
    this.biases = new Array(K).fill(0);

    const X = texts.map(t => textToTfIdf(t, this.vocab, this.idf));
    const Y = labels.map(l => LABEL_INDEX[l]);

    for (let epoch = 0; epoch < epochs; epoch++) {
      const dW = Array.from({ length: K }, () => new Array(V).fill(0));
      const dB = new Array(K).fill(0);

      for (let i = 0; i < X.length; i++) {
        const x = X[i];
        const scores = this.weights.map((w, k) => dotProduct(w, x) + this.biases[k]);
        const probs = softmax(scores);
        for (let k = 0; k < K; k++) {
          const diff = probs[k] - (Y[i] === k ? 1 : 0);
          for (let j = 0; j < V; j++) {
            dW[k][j] += diff * x[j];
          }
          dB[k] += diff;
        }
      }

      const n = X.length;
      for (let k = 0; k < K; k++) {
        for (let j = 0; j < V; j++) {
          this.weights[k][j] -= lr * (dW[k][j] / n + lambda * this.weights[k][j]);
        }
        this.biases[k] -= lr * (dB[k] / n);
      }
    }
    this.trained = true;
  }

  predict(text: string): SentimentResult {
    if (!this.trained) throw new Error("Model not trained");
    const x = textToTfIdf(text, this.vocab, this.idf);
    const scores = this.weights.map((w, k) => dotProduct(w, x) + this.biases[k]);
    const probs = softmax(scores);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: LABELS[maxIdx],
      confidence: Math.round(probs[maxIdx] * 1000) / 1000,
      scores: { positive: probs[0], negative: probs[1], neutral: probs[2] },
      model: "logistic_regression",
      processedText: preprocessText(text),
    };
  }

  isTrained(): boolean { return this.trained; }
}

export class SVMModel {
  private supportVectors: Array<{ x: number[]; label: SentimentLabel; alpha: number }> = [];
  private weights: Record<SentimentLabel, number[]> = {
    positive: [], negative: [], neutral: []
  };
  private biases: Record<SentimentLabel, number> = { positive: 0, negative: 0, neutral: 0 };
  private vocab: string[] = [];
  private idf: Record<string, number> = {};
  private trained = false;

  train(texts: string[], labels: SentimentLabel[]): void {
    this.vocab = buildVocabulary(texts);
    this.idf = computeIdf(texts, this.vocab);
    const V = this.vocab.length;
    const lr = 0.01;
    const epochs = 200;
    const C = 1.0;

    const X = texts.map(t => normalize(textToTfIdf(t, this.vocab, this.idf)));

    for (const targetLabel of LABELS) {
      const w = new Array(V).fill(0);
      let b = 0;
      const Y = labels.map(l => (l === targetLabel ? 1 : -1));

      for (let epoch = 0; epoch < epochs; epoch++) {
        const currentLr = lr / (1 + 0.001 * epoch);
        for (let i = 0; i < X.length; i++) {
          const margin = Y[i] * (dotProduct(w, X[i]) + b);
          if (margin < 1) {
            for (let j = 0; j < V; j++) {
              w[j] = w[j] * (1 - currentLr) + currentLr * C * Y[i] * X[i][j];
            }
            b += currentLr * C * Y[i];
          } else {
            for (let j = 0; j < V; j++) {
              w[j] = w[j] * (1 - currentLr);
            }
          }
        }
      }
      this.weights[targetLabel] = w;
      this.biases[targetLabel] = b;
    }
    this.trained = true;
  }

  predict(text: string): SentimentResult {
    if (!this.trained) throw new Error("Model not trained");
    const x = normalize(textToTfIdf(text, this.vocab, this.idf));
    const rawScores = LABELS.map(label =>
      dotProduct(this.weights[label], x) + this.biases[label]
    );
    const probs = softmax(rawScores.map(s => s * 2));
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: LABELS[maxIdx],
      confidence: Math.round(probs[maxIdx] * 1000) / 1000,
      scores: { positive: probs[0], negative: probs[1], neutral: probs[2] },
      model: "svm",
      processedText: preprocessText(text),
    };
  }

  isTrained(): boolean { return this.trained; }
}

export function computeMetrics(
  predictions: SentimentLabel[],
  actuals: SentimentLabel[],
  modelName: ModelName,
  trainingSamples: number,
  trainingTimeMs: number
): ModelMetrics {
  const matrix: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < actuals.length; i++) {
    const actualIdx = LABEL_INDEX[actuals[i]];
    const predIdx = LABEL_INDEX[predictions[i]];
    matrix[actualIdx][predIdx]++;
  }

  const classReport: Record<string, { precision: number; recall: number; f1Score: number; support: number }> = {};
  let macroP = 0, macroR = 0, macroF1 = 0;
  let correct = 0;

  for (let k = 0; k < LABELS.length; k++) {
    const label = LABELS[k];
    const tp = matrix[k][k];
    const fp = matrix.reduce((s, row, i) => (i !== k ? s + row[k] : s), 0);
    const fn = matrix[k].reduce((s, v, j) => (j !== k ? s + v : s), 0);
    const support = matrix[k].reduce((s, v) => s + v, 0);
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    correct += tp;
    macroP += precision;
    macroR += recall;
    macroF1 += f1;
    classReport[label] = {
      precision: Math.round(precision * 1000) / 1000,
      recall: Math.round(recall * 1000) / 1000,
      f1Score: Math.round(f1 * 1000) / 1000,
      support,
    };
  }

  const n = LABELS.length;
  return {
    model: modelName,
    accuracy: Math.round((correct / actuals.length) * 1000) / 1000,
    precision: Math.round((macroP / n) * 1000) / 1000,
    recall: Math.round((macroR / n) * 1000) / 1000,
    f1Score: Math.round((macroF1 / n) * 1000) / 1000,
    confusionMatrix: { matrix, labels: LABELS },
    classReport,
    trainingSamples,
    testSamples: actuals.length,
    trainingTimeMs: Math.round(trainingTimeMs),
  };
}

export function trainTestSplit<T>(
  data: T[],
  labels: SentimentLabel[],
  testRatio = 0.2
): { trainData: T[]; trainLabels: SentimentLabel[]; testData: T[]; testLabels: SentimentLabel[] } {
  const combined = data.map((d, i) => ({ d, l: labels[i] }));
  for (let i = combined.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [combined[i], combined[j]] = [combined[j], combined[i]];
  }
  const splitIdx = Math.floor(combined.length * (1 - testRatio));
  const train = combined.slice(0, splitIdx);
  const test = combined.slice(splitIdx);
  return {
    trainData: train.map(x => x.d),
    trainLabels: train.map(x => x.l),
    testData: test.map(x => x.d),
    testLabels: test.map(x => x.l),
  };
}
