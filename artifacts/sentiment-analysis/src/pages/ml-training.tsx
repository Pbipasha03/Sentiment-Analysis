import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import {
  getGetModelMetricsQueryKey,
  useAnalyzeSentiment,
  useGetModelMetrics,
  useTrainModels,
  type AnalyzeTextResponse,
  type ModelMetrics,
  type ModelName,
  type SentimentLabel,
} from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, BarChart3, CheckCircle2, Download, Loader2, Upload, Zap } from "lucide-react";

const MODEL_LABELS: Record<ModelName, string> = {
  naive_bayes: "Naive Bayes",
  logistic_regression: "Logistic Regression",
  svm: "SVM",
};

const VALID_LABELS = new Set<SentimentLabel>(["positive", "negative", "neutral"]);

function formatModelName(modelName: ModelName): string {
  return MODEL_LABELS[modelName] ?? modelName;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current.trim());
  return values.map((value) => value.replace(/^"(.*)"$/, "$1").trim());
}

async function parseTrainingFile(file: File): Promise<{ texts: string[]; labels: SentimentLabel[] }> {
  const content = await file.text();
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("CSV must include a header row and at least one data row.");
  }

  const headers = parseCsvLine(lines[0]).map((header) => header.toLowerCase().trim());
  const textIndex = headers.findIndex((header) => ["text", "tweet", "content", "sentence", "review"].includes(header));
  const labelIndex = headers.findIndex((header) => ["sentiment", "label", "target", "class"].includes(header));

  if (textIndex === -1 || labelIndex === -1) {
    throw new Error("CSV must contain 'text' and 'sentiment' columns.");
  }

  const texts: string[] = [];
  const labels: SentimentLabel[] = [];

  for (const line of lines.slice(1)) {
    const columns = parseCsvLine(line);
    const text = (columns[textIndex] ?? "").trim();
    const rawLabel = (columns[labelIndex] ?? "").trim().toLowerCase() as SentimentLabel;

    if (!text) continue;
    if (!VALID_LABELS.has(rawLabel)) {
      throw new Error(`Invalid sentiment label '${columns[labelIndex] ?? ""}'. Use positive, negative, or neutral.`);
    }

    texts.push(text);
    labels.push(rawLabel);
  }

  if (texts.length < 10) {
    throw new Error("Need at least 10 valid rows to train models.");
  }

  return { texts, labels };
}

function downloadTextFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.URL.revokeObjectURL(url);
}

function buildMetricsCsv(metrics: ModelMetrics[]): string {
  const header = "model,accuracy,precision,recall,f1_score,training_samples,test_samples,training_time_ms";
  const rows = metrics.map((metric) =>
    [
      metric.model,
      metric.accuracy,
      metric.precision,
      metric.recall,
      metric.f1Score,
      metric.trainingSamples,
      metric.testSamples,
      metric.trainingTimeMs,
    ].join(","),
  );
  return [header, ...rows].join("\n");
}

export default function MLTraining() {
  const queryClient = useQueryClient();
  const { data: metricsData, isLoading: metricsLoading } = useGetModelMetrics();
  const trainMutation = useTrainModels();
  const analyzeMutation = useAnalyzeSentiment();

  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [predictText, setPredictText] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelName>("naive_bayes");

  const prediction: AnalyzeTextResponse | undefined = analyzeMutation.data;
  const trained = Boolean(metricsData?.trained);
  const modelMetrics = metricsData?.metrics ?? [];

  useEffect(() => {
    if (metricsData?.bestModel) {
      setSelectedModel(metricsData.bestModel);
    }
  }, [metricsData?.bestModel]);

  const summary = useMemo(() => {
    if (!metricsData?.trained || modelMetrics.length === 0) return null;

    const firstMetric = modelMetrics[0];
    return {
      totalSamples: firstMetric.trainingSamples + firstMetric.testSamples,
      trainSamples: firstMetric.trainingSamples,
      testSamples: firstMetric.testSamples,
      classCount: firstMetric.confusionMatrix.labels.length,
      trainingTimeMs: modelMetrics.reduce((sum, metric) => sum + metric.trainingTimeMs, 0),
    };
  }, [metricsData?.trained, modelMetrics]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0] ?? null;
    setError(null);
    setFile(selectedFile);
  };

  const handleTrain = async () => {
    setError(null);

    try {
      const payload = file
        ? await parseTrainingFile(file)
        : { useDefaultDataset: true as const };

      trainMutation.mutate(
        { data: payload },
        {
          onSuccess: (data) => {
            queryClient.setQueryData(getGetModelMetricsQueryKey(), {
              trained: true,
              metrics: data.metrics,
              bestModel: data.bestModel,
              lastTrainedAt: new Date().toISOString(),
            });
            queryClient.invalidateQueries({ queryKey: getGetModelMetricsQueryKey() });
            setSelectedModel(data.bestModel);
            setFile(null);
          },
          onError: (mutationError) => {
            setError(getErrorMessage(mutationError));
          },
        },
      );
    } catch (caughtError) {
      setError(getErrorMessage(caughtError));
    }
  };

  const handlePredict = () => {
    if (!predictText.trim()) {
      setError("Please enter text to analyze.");
      return;
    }

    setError(null);
    analyzeMutation.mutate(
      { data: { text: predictText.trim(), model: selectedModel } },
      {
        onError: (mutationError) => {
          setError(getErrorMessage(mutationError));
        },
      },
    );
  };

  const handleDownloadReport = (format: "csv" | "json") => {
    if (modelMetrics.length === 0) return;

    if (format === "csv") {
      downloadTextFile("model_metrics.csv", buildMetricsCsv(modelMetrics), "text/csv;charset=utf-8");
      return;
    }

    downloadTextFile(
      "model_metrics.json",
      JSON.stringify(
        {
          trained: metricsData?.trained ?? false,
          bestModel: metricsData?.bestModel ?? null,
          lastTrainedAt: metricsData?.lastTrainedAt ?? null,
          metrics: modelMetrics,
        },
        null,
        2,
      ),
      "application/json;charset=utf-8",
    );
  };

  if (metricsLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-4xl font-bold tracking-tight">ML Model Training Pipeline</h1>
        <p className="text-muted-foreground mt-2">
          Train the shared sentiment models and keep the whole app in sync.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="border-2">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Upload className="mr-2 h-5 w-5" /> Step 1: Upload CSV or Use Default Dataset
          </CardTitle>
          <CardDescription>
            Optional CSV columns: <span className="font-mono">text</span> and <span className="font-mono">sentiment</span>. If no file is selected, the default sample dataset is used.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="border-2 border-dashed rounded-lg p-8 text-center">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="csv-input"
            />
            <label htmlFor="csv-input" className="cursor-pointer">
              <div className="space-y-2">
                <Upload className="mx-auto h-8 w-8 text-muted-foreground" />
                <div>
                  <p className="font-medium">{file ? `Selected: ${file.name}` : "Click to select CSV file"}</p>
                  <p className="text-sm text-muted-foreground">Leave empty to train the shared default models</p>
                </div>
              </div>
            </label>
          </div>

          <div className="rounded-lg bg-muted/40 p-3 text-sm text-muted-foreground">
            Shared training status: <span className="font-semibold text-foreground">{trained ? "Trained" : "Not trained"}</span>
          </div>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleTrain}
            disabled={trainMutation.isPending}
            size="lg"
            className="w-full"
          >
            {trainMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training Models...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" /> {trained ? "Retrain Models" : "Train Models"}
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      {trained && metricsData && summary && (
        <>
          <Card className="bg-gradient-to-r from-green-50 to-blue-50">
            <CardHeader>
              <CardTitle className="flex items-center">
                <CheckCircle2 className="mr-2 h-5 w-5 text-green-600" /> Training Complete
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <p className="text-sm text-muted-foreground">Total Samples</p>
                <p className="text-2xl font-bold">{summary.totalSamples}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Train/Test Split</p>
                <p className="text-2xl font-bold">{summary.trainSamples}/{summary.testSamples}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Classes</p>
                <p className="text-2xl font-bold">{summary.classCount}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Best Model</p>
                <p className="text-2xl font-bold">{formatModelName(metricsData.bestModel ?? "naive_bayes")}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="mr-2 h-5 w-5" /> Model Comparison
              </CardTitle>
              <CardDescription>Performance metrics for the shared trained models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b">
                    <tr className="bg-muted/50">
                      <th className="p-2 text-left font-semibold">Model</th>
                      <th className="p-2 text-center font-semibold">Accuracy</th>
                      <th className="p-2 text-center font-semibold">Precision</th>
                      <th className="p-2 text-center font-semibold">Recall</th>
                      <th className="p-2 text-center font-semibold">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelMetrics.map((metric) => (
                      <tr key={metric.model} className="border-b hover:bg-muted/30">
                        <td className="p-2">
                          <div className="flex items-center gap-2">
                            {formatModelName(metric.model)}
                            {metric.model === metricsData.bestModel && (
                              <Badge className="bg-yellow-500 text-black hover:bg-yellow-500">Best</Badge>
                            )}
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div className="font-bold text-green-600">
                            {(metric.accuracy * 100).toFixed(1)}%
                          </div>
                          <Progress value={metric.accuracy * 100} className="mt-1" />
                        </td>
                        <td className="p-2 text-center">{(metric.precision * 100).toFixed(1)}%</td>
                        <td className="p-2 text-center">{(metric.recall * 100).toFixed(1)}%</td>
                        <td className="p-2 text-center">{(metric.f1Score * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Per-Class Metrics</CardTitle>
              <CardDescription>Detailed performance for each sentiment class</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {modelMetrics.map((metric) => (
                <div key={metric.model} className="space-y-3 rounded-lg border p-4">
                  <h3 className="text-lg font-semibold">{formatModelName(metric.model)}</h3>
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                    {Object.entries(metric.classReport).map(([className, classMetrics]) => (
                      <div key={className} className="rounded bg-muted/50 p-3">
                        <p className="mb-2 text-sm font-medium capitalize">{className}</p>
                        <div className="space-y-1 text-xs">
                          <p>P: {(classMetrics.precision * 100).toFixed(1)}%</p>
                          <p>R: {(classMetrics.recall * 100).toFixed(1)}%</p>
                          <p>F1: {(classMetrics.f1Score * 100).toFixed(1)}%</p>
                          <p className="text-muted-foreground">samples: {classMetrics.support}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Download className="mr-2 h-5 w-5" /> Download Report
              </CardTitle>
            </CardHeader>
            <CardContent className="flex gap-4">
              <Button onClick={() => handleDownloadReport("csv")} variant="outline" className="flex-1">
                Download as CSV
              </Button>
              <Button onClick={() => handleDownloadReport("json")} variant="outline" className="flex-1">
                Download as JSON
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Test Predictions</CardTitle>
              <CardDescription>Analyze new text using the same shared trained models</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Model</label>
                <div className="flex gap-2">
                  {modelMetrics.map((metric) => (
                    <Button
                      key={metric.model}
                      variant={selectedModel === metric.model ? "default" : "outline"}
                      onClick={() => setSelectedModel(metric.model)}
                      size="sm"
                    >
                      {formatModelName(metric.model)}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="predict-text" className="text-sm font-medium">
                  Enter Text to Analyze
                </label>
                <Textarea
                  id="predict-text"
                  placeholder="Type your text here..."
                  value={predictText}
                  onChange={(event) => setPredictText(event.target.value)}
                  className="min-h-[100px]"
                />
              </div>

              <Button
                onClick={handlePredict}
                disabled={!predictText.trim() || analyzeMutation.isPending || !trained}
                className="w-full"
              >
                {analyzeMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...
                  </>
                ) : (
                  "Analyze Sentiment"
                )}
              </Button>
            </CardContent>

            {prediction && (
              <CardFooter className="flex-col items-start border-t pt-4">
                <div className="w-full space-y-3">
                  <div>
                    <p className="mb-1 text-sm text-muted-foreground">Prediction</p>
                    <Badge className="px-3 py-1 text-lg">
                      {prediction.result.label.toUpperCase()}
                    </Badge>
                  </div>

                  <div>
                    <p className="mb-2 text-sm text-muted-foreground">Confidence Scores</p>
                    {Object.entries(prediction.result.scores).map(([label, score]) => (
                      <div key={label} className="mb-2 flex items-center gap-2">
                        <span className="w-24 text-sm font-medium capitalize">{label}</span>
                        <div className="flex-1">
                          <Progress value={score * 100} />
                        </div>
                        <span className="text-sm font-semibold">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>

                  <p className="text-xs text-muted-foreground">
                    Model: {formatModelName(prediction.result.model)}
                  </p>
                </div>
              </CardFooter>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
