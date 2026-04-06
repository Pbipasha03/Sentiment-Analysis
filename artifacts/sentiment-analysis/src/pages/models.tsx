import { useGetModelMetrics, useTrainModels } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, RefreshCw } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useQueryClient } from "@tanstack/react-query";
import { getGetModelMetricsQueryKey } from "@workspace/api-client-react";

export default function Models() {
  const queryClient = useQueryClient();
  const { data: metricsData, isLoading } = useGetModelMetrics();
  const trainMutation = useTrainModels();

  const handleTrain = () => {
    trainMutation.mutate(
      { data: { useDefaultDataset: true } },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: getGetModelMetricsQueryKey() });
        }
      }
    );
  };

  const getAccuracyData = () => {
    if (!metricsData?.metrics) return [];
    return metricsData.metrics.map(m => ({
      name: m.model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      accuracy: Number((m.accuracy * 100).toFixed(2)),
      f1: Number((m.f1Score * 100).toFixed(2))
    }));
  };

  if (isLoading) {
    return <div className="flex h-64 items-center justify-center"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Model Performance</h1>
          <p className="text-muted-foreground mt-2">
            Evaluate classification metrics across implemented architectures.
          </p>
        </div>
        <Button 
          onClick={handleTrain} 
          disabled={trainMutation.isPending}
          variant={metricsData?.trained ? "outline" : "default"}
        >
          {trainMutation.isPending ? (
            <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training...</>
          ) : (
            <><RefreshCw className="mr-2 h-4 w-4" /> {metricsData?.trained ? 'Retrain Models' : 'Train Models'}</>
          )}
        </Button>
      </div>

      {!metricsData?.trained ? (
        <Card className="border-dashed border-2 bg-muted/20">
          <CardContent className="flex flex-col items-center justify-center h-64 space-y-4">
            <div className="text-center space-y-2">
              <h3 className="text-xl font-semibold">Models Untrained</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                The NLP models need to be trained on the sample dataset before they can generate performance metrics or run inference.
              </p>
            </div>
            <Button onClick={handleTrain} disabled={trainMutation.isPending}>
              {trainMutation.isPending ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training (This may take a minute)</>
              ) : (
                "Train Models Now"
              )}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-6 md:grid-cols-12">
            <Card className="md:col-span-8">
              <CardHeader>
                <CardTitle>Comparative Performance</CardTitle>
                <CardDescription>Accuracy and F1 Score comparison across architectures.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getAccuracyData()} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                      <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}%`} />
                      <RechartsTooltip 
                        cursor={{ fill: 'hsl(var(--muted))', opacity: 0.4 }}
                        contentStyle={{ borderRadius: '8px', border: '1px solid hsl(var(--border))' }}
                      />
                      <Legend />
                      <Bar dataKey="accuracy" name="Accuracy" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} maxBarSize={50} />
                      <Bar dataKey="f1" name="F1 Score" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} maxBarSize={50} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="md:col-span-4">
              <CardHeader>
                <CardTitle>System Information</CardTitle>
                <CardDescription>Training metadata</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <p className="text-sm font-medium text-muted-foreground mb-1">Best Performing Model</p>
                  <Badge variant="default" className="text-sm py-1 px-3">
                    {metricsData.bestModel?.replace('_', ' ').toUpperCase()}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-muted-foreground">Train Samples</p>
                    <p className="text-2xl font-bold">{metricsData.metrics[0]?.trainingSamples.toLocaleString()}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-muted-foreground">Test Samples</p>
                    <p className="text-2xl font-bold">{metricsData.metrics[0]?.testSamples.toLocaleString()}</p>
                  </div>
                </div>

                <div className="space-y-1 pt-4 border-t">
                  <p className="text-sm font-medium text-muted-foreground">Total Training Time</p>
                  <p className="text-lg">{metricsData.metrics.reduce((acc, m) => acc + m.trainingTimeMs, 0)}ms</p>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Detailed Metrics</CardTitle>
              <CardDescription>Select a model to view deep performance characteristics.</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue={metricsData.metrics[0]?.model} className="w-full">
                <TabsList className="mb-6">
                  {metricsData.metrics.map(m => (
                    <TabsTrigger key={m.model} value={m.model}>
                      {m.model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </TabsTrigger>
                  ))}
                </TabsList>
                
                {metricsData.metrics.map(m => (
                  <TabsContent key={m.model} value={m.model} className="space-y-8">
                    <div className="grid md:grid-cols-2 gap-8">
                      <div>
                        <h3 className="text-lg font-semibold mb-4">Classification Report</h3>
                        <div className="rounded-md border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Class</TableHead>
                                <TableHead className="text-right">Precision</TableHead>
                                <TableHead className="text-right">Recall</TableHead>
                                <TableHead className="text-right">F1-Score</TableHead>
                                <TableHead className="text-right">Support</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {Object.entries(m.classReport).map(([className, scores]) => (
                                className !== 'accuracy' && className !== 'macro avg' && className !== 'weighted avg' && (
                                  <TableRow key={className}>
                                    <TableCell className="font-medium capitalize">{className}</TableCell>
                                    <TableCell className="text-right">{(scores.precision * 100).toFixed(1)}%</TableCell>
                                    <TableCell className="text-right">{(scores.recall * 100).toFixed(1)}%</TableCell>
                                    <TableCell className="text-right">{(scores.f1Score * 100).toFixed(1)}%</TableCell>
                                    <TableCell className="text-right">{scores.support}</TableCell>
                                  </TableRow>
                                )
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>

                      <div>
                        <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
                        <div className="grid grid-cols-[auto_1fr] gap-2">
                          <div className="flex flex-col justify-end pb-8 pr-2">
                            <span className="text-xs font-semibold text-muted-foreground rotate-180" style={{ writingMode: 'vertical-rl' }}>True Label</span>
                          </div>
                          <div>
                            <div className="grid gap-1 mb-2" style={{ gridTemplateColumns: `repeat(${m.confusionMatrix.labels.length}, 1fr)` }}>
                              {m.confusionMatrix.labels.map((label, idx) => (
                                <div key={idx} className="text-center text-xs font-semibold text-muted-foreground truncate">{label.substring(0, 3)}</div>
                              ))}
                            </div>
                            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${m.confusionMatrix.labels.length}, 1fr)` }}>
                              {m.confusionMatrix.matrix.map((row, i) => 
                                row.map((val, j) => {
                                  // Max value calculation for color intensity
                                  const flatMatrix = m.confusionMatrix.matrix.flat();
                                  const max = Math.max(...flatMatrix);
                                  const intensity = Math.max(0.1, val / max);
                                  const isCorrect = i === j;
                                  
                                  return (
                                    <div 
                                      key={`${i}-${j}`} 
                                      className={`aspect-square flex items-center justify-center rounded-sm text-sm font-medium ${intensity > 0.5 ? 'text-white' : 'text-foreground'}`}
                                      style={{ 
                                        backgroundColor: isCorrect 
                                          ? `rgba(34, 197, 94, ${intensity})` 
                                          : `rgba(239, 68, 68, ${intensity})`,
                                        border: '1px solid hsl(var(--border))'
                                      }}
                                      title={`True: ${m.confusionMatrix.labels[i]}, Pred: ${m.confusionMatrix.labels[j]}, Count: ${val}`}
                                    >
                                      {val}
                                    </div>
                                  )
                                })
                              )}
                            </div>
                            <div className="text-center mt-2">
                              <span className="text-xs font-semibold text-muted-foreground">Predicted Label</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}