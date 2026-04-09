import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Upload, Loader2, CheckCircle2, BarChart3, Download, Zap } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';

export default function MLTraining() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trained, setTrained] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictText, setPredictText] = useState('');
  const [predictLoading, setPredictLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);

  const API_BASE = 'http://127.0.0.1:5001/api/ml';

  // ─── FILE UPLOAD & TRAINING ───────────────────────────────────────────

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid CSV file');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setMetrics(data);
      setTrained(true);
      setSelectedModel(data.best_model);
      setFile(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ─── PREDICTION ───────────────────────────────────────────────────────

  const handlePredict = async () => {
    if (!predictText.trim()) {
      setError('Please enter text to analyze');
      return;
    }

    setPredictLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: predictText,
          model: selectedModel,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredictLoading(false);
    }
  };

  // ─── DOWNLOAD REPORT ──────────────────────────────────────────────────

  const downloadReport = async (format) => {
    try {
      const response = await fetch(`${API_BASE}/report/${format}`);
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report.${format}`;
      a.click();
    } catch (err) {
      setError(err.message);
    }
  };

  // ─── RENDER ───────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight">ML Model Training Pipeline</h1>
        <p className="text-muted-foreground mt-2">
          Upload CSV, train models, and analyze sentiment with Naive Bayes, Logistic Regression, and SVM
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* SECTION 1: CSV UPLOAD & TRAINING */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Upload className="mr-2 h-5 w-5" /> Step 1: Upload CSV
          </CardTitle>
          <CardDescription>
            CSV must have 'text' and 'sentiment' columns
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
                  <p className="font-medium">
                    {file ? '✓ ' + file.name : 'Click to select CSV file'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or drag and drop
                  </p>
                </div>
              </div>
            </label>
          </div>

          {file && (
            <div className="bg-blue-50 p-3 rounded text-sm text-blue-900">
              ✓ File ready: <span className="font-mono">{file.name}</span>
            </div>
          )}
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleUpload}
            disabled={!file || loading}
            size="lg"
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training Models...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" /> Train Models
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      {/* SECTION 2: MODEL METRICS (if trained) */}
      {trained && metrics && (
        <>
          {/* Summary */}
          <Card className="bg-gradient-to-r from-green-50 to-blue-50">
            <CardHeader>
              <CardTitle className="flex items-center">
                <CheckCircle2 className="mr-2 h-5 w-5 text-green-600" /> Training Complete!
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Total Samples</p>
                <p className="text-2xl font-bold">{metrics.data_statistics.total_samples}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Train/Test Split</p>
                <p className="text-2xl font-bold">{metrics.data_statistics.train_samples}/{metrics.data_statistics.test_samples}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Features (TF-IDF)</p>
                <p className="text-2xl font-bold">{metrics.data_statistics.features}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Classes</p>
                <p className="text-2xl font-bold">{metrics.data_statistics.classes.length}</p>
              </div>
            </CardContent>
          </Card>

          {/* Model Comparison Table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="mr-2 h-5 w-5" /> Model Comparison
              </CardTitle>
              <CardDescription>
                Performance metrics for all trained models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b">
                    <tr className="bg-muted/50">
                      <th className="text-left p-2 font-semibold">Model</th>
                      <th className="text-center p-2 font-semibold">Accuracy</th>
                      <th className="text-center p-2 font-semibold">Precision</th>
                      <th className="text-center p-2 font-semibold">Recall</th>
                      <th className="text-center p-2 font-semibold">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.models).map(([modelName, modelMetrics]) => (
                      <tr key={modelName} className="border-b hover:bg-muted/30">
                        <td className="p-2">
                          <div className="flex items-center gap-2">
                            {modelName}
                            {modelName === metrics.best_model && (
                              <Badge className="bg-yellow-500">🏆 Best</Badge>
                            )}
                          </div>
                        </td>
                        <td className="text-center p-2">
                          <div className="font-bold text-green-600">
                            {(modelMetrics.accuracy * 100).toFixed(1)}%
                          </div>
                          <Progress value={modelMetrics.accuracy * 100} className="mt-1" />
                        </td>
                        <td className="text-center p-2">
                          {(modelMetrics.precision * 100).toFixed(1)}%
                        </td>
                        <td className="text-center p-2">
                          {(modelMetrics.recall * 100).toFixed(1)}%
                        </td>
                        <td className="text-center p-2">
                          {(modelMetrics.f1 * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Confusion Matrices */}
          <Card>
            <CardHeader>
              <CardTitle>Per-Class Metrics</CardTitle>
              <CardDescription>
                Detailed performance for each sentiment class
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {Object.entries(metrics.models).map(([modelName, modelMetrics]) => (
                <div key={modelName} className="border rounded-lg p-4 space-y-3">
                  <h3 className="font-semibold text-lg">{modelName}</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(modelMetrics.per_class).map(([className, classMetrics]) => (
                      <div key={className} className="bg-muted/50 p-3 rounded">
                        <p className="text-sm font-medium capitalize mb-2">{className}</p>
                        <div className="text-xs space-y-1">
                          <p>P: {(classMetrics.precision * 100).toFixed(1)}%</p>
                          <p>R: {(classMetrics.recall * 100).toFixed(1)}%</p>
                          <p>F1: {(classMetrics.f1 * 100).toFixed(1)}%</p>
                          <p className="text-muted-foreground">samples: {classMetrics.support}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Download Reports */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Download className="mr-2 h-5 w-5" /> Download Report
              </CardTitle>
            </CardHeader>
            <CardContent className="flex gap-4">
              <Button
                onClick={() => downloadReport('csv')}
                variant="outline"
                className="flex-1"
              >
                📊 Download as CSV
              </Button>
              <Button
                onClick={() => downloadReport('json')}
                variant="outline"
                className="flex-1"
              >
                📋 Download as JSON
              </Button>
            </CardContent>
          </Card>

          {/* SECTION 3: PREDICTION */}
          <Card>
            <CardHeader>
              <CardTitle>Test Predictions</CardTitle>
              <CardDescription>
                Analyze new text with trained models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Model</label>
                <div className="flex gap-2">
                  {Object.keys(metrics.models).map((modelName) => (
                    <Button
                      key={modelName}
                      variant={selectedModel === modelName ? 'default' : 'outline'}
                      onClick={() => setSelectedModel(modelName)}
                      size="sm"
                    >
                      {modelName}
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
                  onChange={(e) => setPredictText(e.target.value)}
                  className="min-h-[100px]"
                />
              </div>

              <Button
                onClick={handlePredict}
                disabled={!predictText.trim() || predictLoading}
                className="w-full"
              >
                {predictLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...
                  </>
                ) : (
                  'Analyze Sentiment'
                )}
              </Button>
            </CardContent>

            {prediction && (
              <CardFooter className="flex-col items-start border-t pt-4">
                <div className="w-full space-y-3">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Prediction</p>
                    <Badge className="text-lg px-3 py-1">
                      {prediction.prediction.toUpperCase()}
                    </Badge>
                  </div>

                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Confidence Scores</p>
                    {Object.entries(prediction.scores).map(([label, score]) => (
                      <div key={label} className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-medium capitalize w-24">{label}</span>
                        <div className="flex-1">
                          <Progress value={score * 100} />
                        </div>
                        <span className="text-sm font-semibold">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>

                  <p className="text-xs text-muted-foreground">
                    Model: {prediction.model}
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
