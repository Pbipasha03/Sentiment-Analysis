import { useCompareModels, useGetModelMetrics } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Loader2, GitCompare, CheckCircle2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function Compare() {
  const [text, setText] = useState("");
  
  const { data: metricsData } = useGetModelMetrics();
  const compareMutation = useCompareModels();

  const handleCompare = () => {
    if (!text.trim()) return;
    compareMutation.mutate({ data: { text } });
  };

  const isTrained = metricsData?.trained;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Comparison</h1>
        <p className="text-muted-foreground mt-2">
          Run the same input through calibrated models and review the final sentiment.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Input Context</CardTitle>
          <CardDescription>Enter a sentence to check whether the tone is positive, negative, or neutral.</CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea 
            placeholder="Type a sentence with complex sentiment..." 
            className="min-h-[100px] resize-y"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </CardContent>
        <CardFooter className="flex justify-between items-center bg-muted/30 py-4 border-t">
          <div className="text-sm text-muted-foreground">
            {compareMutation.data && compareMutation.data.agreement === 1 && (
              <span className="flex items-center text-green-600 font-medium">
                <CheckCircle2 className="w-4 h-4 mr-1" /> Unanimous agreement
              </span>
            )}
            {compareMutation.data && compareMutation.data.agreement < 1 && (
              <span className="flex items-center text-amber-600 font-medium">
                <GitCompare className="w-4 h-4 mr-1" /> Partial disagreement
              </span>
            )}
          </div>
          <Button 
            onClick={handleCompare} 
            disabled={!text.trim() || compareMutation.isPending || !isTrained}
          >
            {compareMutation.isPending ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Evaluating</>
            ) : !isTrained ? (
              "Models not trained"
            ) : (
              <><GitCompare className="mr-2 h-4 w-4" /> Compare Models</>
            )}
          </Button>
        </CardFooter>
      </Card>

      {compareMutation.data && (
        <div className="grid gap-6 md:grid-cols-3">
          {compareMutation.data.comparisons.map((result) => (
            <Card key={result.model} className="flex flex-col">
              <CardHeader className="bg-muted/30 pb-4 border-b">
                <CardTitle className="text-lg">
                  {result.model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </CardTitle>
                {metricsData?.bestModel === result.model && (
                  <Badge className="absolute top-4 right-4" variant="secondary">Best Model</Badge>
                )}
              </CardHeader>
              <CardContent className="pt-6 flex-1 space-y-6">
                <div className="flex flex-col items-center justify-center space-y-2 pb-4 border-b">
                  <span className="text-sm text-muted-foreground font-medium uppercase tracking-wider">Prediction</span>
                  <Badge 
                    variant="outline" 
                    className={`text-xl py-2 px-6 ${
                      result.label === 'positive' ? 'bg-green-500/10 text-green-700 border-green-500/20' :
                      result.label === 'negative' ? 'bg-red-500/10 text-red-700 border-red-500/20' :
                      'bg-amber-500/10 text-amber-700 border-amber-500/20'
                    }`}
                  >
                    {result.label.toUpperCase()}
                  </Badge>
                </div>

                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-sm">
                      <span className="text-green-600 font-medium">Pos</span>
                      <span className="text-muted-foreground text-xs">{(result.scores.positive * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={result.scores.positive * 100} className="h-1.5 bg-green-100 [&>div]:bg-green-500" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-sm">
                      <span className="text-amber-600 font-medium">Neu</span>
                      <span className="text-muted-foreground text-xs">{(result.scores.neutral * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={result.scores.neutral * 100} className="h-1.5 bg-amber-100 [&>div]:bg-amber-500" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-sm">
                      <span className="text-red-600 font-medium">Neg</span>
                      <span className="text-muted-foreground text-xs">{(result.scores.negative * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={result.scores.negative * 100} className="h-1.5 bg-red-100 [&>div]:bg-red-500" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
