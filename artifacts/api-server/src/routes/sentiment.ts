import { Router, type IRouter } from "express";
import {
  AnalyzeSentimentBody,
  AnalyzeSentimentResponse,
  AnalyzeBatchBody,
  AnalyzeBatchResponse,
  GetSampleDatasetQueryParams,
  GetSampleDatasetResponse,
  TrainModelsBody,
  TrainModelsResponse,
  GetModelMetricsResponse,
  CompareModelsBody,
  CompareModelsResponse,
  GenerateReportBody,
  GenerateReportResponse,
  GetWordCloudDataBody,
  GetWordCloudDataResponse,
} from "@workspace/api-zod";
import { getModelStore, trainAllModels, getModel, ensureDefaultModelsTrained } from "../lib/modelStore.js";
import { getDefaultDataset, SAMPLE_DATASET } from "../lib/sampleDataset.js";
import { extractKeywords, getWordFrequencies } from "../lib/nlpUtils.js";
import type { SentimentLabel, ModelName } from "../lib/mlModels.js";

const router: IRouter = Router();

router.post("/sentiment/analyze", async (req, res): Promise<void> => {
  const parsed = AnalyzeSentimentBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  await ensureDefaultModelsTrained();
  const store = getModelStore();
  const { text, model: modelName } = parsed.data;

  const chosenModel = modelName ?? store.bestModel;
  const primaryModel = getModel(chosenModel as ModelName);
  const startTime = Date.now();
  const result = primaryModel.predict(text);

  const allModels = (["naive_bayes", "logistic_regression", "svm"] as ModelName[]).map(m =>
    getModel(m).predict(text)
  );

  const keywords = extractKeywords(text, 10);

  res.json(
    AnalyzeSentimentResponse.parse({
      originalText: text,
      result,
      allModels,
      keywords,
      processingTimeMs: Date.now() - startTime,
    })
  );
});

router.post("/sentiment/analyze-batch", async (req, res): Promise<void> => {
  const parsed = AnalyzeBatchBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  await ensureDefaultModelsTrained();
  const store = getModelStore();
  const { texts, model: modelName } = parsed.data;
  const chosenModel = (modelName ?? store.bestModel) as ModelName;
  const model = getModel(chosenModel);
  const startTime = Date.now();

  const results = texts.map((text, index) => {
    const result = model.predict(text);
    return {
      index,
      text,
      label: result.label as SentimentLabel,
      confidence: result.confidence,
      model: chosenModel,
    };
  });

  const summary = {
    total: results.length,
    positive: results.filter(r => r.label === "positive").length,
    negative: results.filter(r => r.label === "negative").length,
    neutral: results.filter(r => r.label === "neutral").length,
    positivePercent: 0,
    negativePercent: 0,
    neutralPercent: 0,
  };
  const total = summary.total || 1;
  summary.positivePercent = Math.round((summary.positive / total) * 1000) / 10;
  summary.negativePercent = Math.round((summary.negative / total) * 1000) / 10;
  summary.neutralPercent = Math.round((summary.neutral / total) * 1000) / 10;

  res.json(
    AnalyzeBatchResponse.parse({
      results,
      summary,
      processingTimeMs: Date.now() - startTime,
    })
  );
});

router.get("/dataset/sample", async (req, res): Promise<void> => {
  const parsed = GetSampleDatasetQueryParams.safeParse(req.query);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  const limit = parsed.data.limit ?? 50;
  const samples = SAMPLE_DATASET.slice(0, limit);

  res.json(
    GetSampleDatasetResponse.parse({
      samples,
      total: SAMPLE_DATASET.length,
      datasetName: "Twitter Sentiment Dataset (Sample)",
    })
  );
});

router.post("/models/train", async (req, res): Promise<void> => {
  const parsed = TrainModelsBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  const startTime = Date.now();
  let texts: string[];
  let labels: SentimentLabel[];

  if (parsed.data.useDefaultDataset || (!parsed.data.texts?.length)) {
    const dataset = getDefaultDataset();
    texts = dataset.texts;
    labels = dataset.labels;
  } else {
    texts = parsed.data.texts ?? [];
    labels = (parsed.data.labels ?? []) as SentimentLabel[];
  }

  if (texts.length < 10) {
    res.status(400).json({ error: "Need at least 10 samples to train models" });
    return;
  }

  const metrics = await trainAllModels(texts, labels);
  const store = getModelStore();

  res.json(
    TrainModelsResponse.parse({
      success: true,
      metrics,
      bestModel: store.bestModel,
      totalTimeMs: Date.now() - startTime,
    })
  );
});

router.get("/models/metrics", async (_req, res): Promise<void> => {
  const store = getModelStore();

  if (!store.trained) {
    res.json(
      GetModelMetricsResponse.parse({
        trained: false,
        metrics: [],
        bestModel: "naive_bayes",
        lastTrainedAt: null,
      })
    );
    return;
  }

  res.json(
    GetModelMetricsResponse.parse({
      trained: store.trained,
      metrics: store.metrics,
      bestModel: store.bestModel,
      lastTrainedAt: store.lastTrainedAt?.toISOString() ?? null,
    })
  );
});

router.post("/models/compare", async (req, res): Promise<void> => {
  const parsed = CompareModelsBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  await ensureDefaultModelsTrained();
  const { text } = parsed.data;

  const comparisons = (["naive_bayes", "logistic_regression", "svm"] as ModelName[]).map(m =>
    getModel(m).predict(text)
  );

  const labelCounts: Record<string, number> = {};
  for (const c of comparisons) {
    labelCounts[c.label] = (labelCounts[c.label] || 0) + 1;
  }
  const consensus = Object.entries(labelCounts).sort((a, b) => b[1] - a[1])[0][0] as SentimentLabel;
  const agreement = Math.max(...Object.values(labelCounts)) / comparisons.length;

  res.json(
    CompareModelsResponse.parse({
      text,
      comparisons,
      consensus,
      agreement: Math.round(agreement * 1000) / 1000,
    })
  );
});

router.post("/report/generate", async (req, res): Promise<void> => {
  const parsed = GenerateReportBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  const { results, format } = parsed.data;
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");

  let data: string;
  let filename: string;

  if (format === "csv") {
    const header = "Index,Text,Sentiment,Confidence,Model";
    const rows = results.map(
      r => `${r.index},"${(r.text ?? "").replace(/"/g, '""')}",${r.label},${r.confidence},${r.model}`
    );
    data = [header, ...rows].join("\n");
    filename = `sentiment_analysis_${timestamp}.csv`;
  } else {
    data = JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        total_records: results.length,
        summary: {
          positive: results.filter(r => r.label === "positive").length,
          negative: results.filter(r => r.label === "negative").length,
          neutral: results.filter(r => r.label === "neutral").length,
        },
        results,
      },
      null,
      2
    );
    filename = `sentiment_analysis_${timestamp}.json`;
  }

  res.json(
    GenerateReportResponse.parse({
      data,
      filename,
      format,
    })
  );
});

router.post("/dataset/wordcloud", async (req, res): Promise<void> => {
  const parsed = GetWordCloudDataBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  const { texts, sentimentFilter } = parsed.data;

  let filteredTexts = texts;

  if (sentimentFilter) {
    await ensureDefaultModelsTrained();
    const store = getModelStore();
    const model = getModel(store.bestModel);
    filteredTexts = texts.filter(t => {
      const result = model.predict(t);
      return result.label === sentimentFilter;
    });
  }

  const words = getWordFrequencies(filteredTexts);

  res.json(GetWordCloudDataResponse.parse({ words }));
});

export default router;
