import { useAnalyzeSentiment, useGetModelMetrics } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Loader2, MessageSquareText } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function Analyze() {
  const [text, setText] = useState("");
  const [model, setModel] = useState<string>("naive_bayes");
  
  const { data: metricsData } = useGetModelMetrics();
  const analyzeMutation = useAnalyzeSentiment();

  const handleAnalyze = () => {
    if (!text.trim()) return;
    analyzeMutation.mutate({ data: { text, model: model as any } });
  };

  const isTrained = metricsData?.trained;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Analyze Text</h1>
        <p className="text-muted-foreground mt-2">
          Run inference on individual text snippets using pre-trained NLP models.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-12">
        <div className="md:col-span-5 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>Enter text to analyze its sentiment.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Text snippet</label>
                <Textarea 
                  placeholder="Type or paste text here..." 
                  className="min-h-[150px] resize-none"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Selection</label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="naive_bayes">Naive Bayes (Fast)</SelectItem>
                    <SelectItem value="logistic_regression">Logistic Regression (Balanced)</SelectItem>
                    <SelectItem value="svm">Support Vector Machine (Accurate)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
            <CardFooter>
              <Button 
                onClick={handleAnalyze} 
                disabled={!text.trim() || analyzeMutation.isPending || !isTrained}
                className="w-full"
              >
                {analyzeMutation.isPending ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing</>
                ) : !isTrained ? (
                  "Models not trained"
                ) : (
                  "Run Analysis"
                )}
              </Button>
            </CardFooter>
          </Card>
        </div>

        <div className="md:col-span-7">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Results</CardTitle>
              <CardDescription>Model prediction and confidence scores.</CardDescription>
            </CardHeader>
            <CardContent>
              {analyzeMutation.data ? (
                <div className="space-y-8">
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold border-b pb-2">Prediction</h3>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Primary Label</span>
                      <Badge 
                        variant="outline" 
                        className={`text-lg py-1 px-4 ${
                          analyzeMutation.data.result.label === 'positive' ? 'bg-green-500/10 text-green-700 border-green-500/20' :
                          analyzeMutation.data.result.label === 'negative' ? 'bg-red-500/10 text-red-700 border-red-500/20' :
                          'bg-amber-500/10 text-amber-700 border-amber-500/20'
                        }`}
                      >
                        {analyzeMutation.data.result.label.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className="font-medium">{(analyzeMutation.data.result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={analyzeMutation.data.result.confidence * 100} className="h-2" />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold border-b pb-2">Class Probabilities</h3>
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-green-600 font-medium">Positive</span>
                          <span className="text-muted-foreground">{(analyzeMutation.data.result.scores.positive * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={analyzeMutation.data.result.scores.positive * 100} className="h-1 bg-green-100 [&>div]:bg-green-500" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-amber-600 font-medium">Neutral</span>
                          <span className="text-muted-foreground">{(analyzeMutation.data.result.scores.neutral * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={analyzeMutation.data.result.scores.neutral * 100} className="h-1 bg-amber-100 [&>div]:bg-amber-500" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-red-600 font-medium">Negative</span>
                          <span className="text-muted-foreground">{(analyzeMutation.data.result.scores.negative * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={analyzeMutation.data.result.scores.negative * 100} className="h-1 bg-red-100 [&>div]:bg-red-500" />
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold border-b pb-2">Extracted Features</h3>
                    <div className="flex flex-wrap gap-2">
                      {analyzeMutation.data.keywords.length > 0 ? (
                        analyzeMutation.data.keywords.map((kw, i) => (
                          <Badge key={i} variant="secondary" className="font-mono">{kw}</Badge>
                        ))
                      ) : (
                        <span className="text-sm text-muted-foreground">No significant keywords extracted</span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <MessageSquareText className="w-12 h-12 opacity-20" />
                  <p>Enter text and run analysis to see results here</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}