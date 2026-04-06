import { useAnalyzeBatch, useGetModelMetrics } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Loader2, UploadCloud, FileText } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from "recharts";

export default function BatchAnalysis() {
  const [model, setModel] = useState<string>("naive_bayes");
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { data: metricsData } = useGetModelMetrics();
  const batchMutation = useAnalyzeBatch();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const csv = event.target?.result as string;
      // Simple CSV parsing (split by newline, skip header if present, handle simple quotes)
      const lines = csv.split('\n').filter(line => line.trim().length > 0);
      
      let texts: string[] = [];
      // Try to determine if first line is header
      const firstLine = lines[0].toLowerCase();
      const hasHeader = firstLine.includes('text') || firstLine.includes('tweet') || firstLine.includes('content');
      
      const startIndex = hasHeader ? 1 : 0;
      
      for (let i = startIndex; i < lines.length; i++) {
        // Very basic CSV parsing - just grab the first column or assume the whole line is text
        let text = lines[i];
        if (text.includes(',')) {
          // Try to extract a quoted string if present, otherwise just use the first column
          const match = text.match(/(?:^|,)(?:"([^"]*)"|([^,]*))/);
          if (match) {
            text = match[1] || match[2] || text;
          }
        }
        if (text.trim()) {
          texts.push(text.trim());
        }
      }

      if (texts.length > 0) {
        batchMutation.mutate({ data: { texts, model: model as any } });
      }
    };
    reader.readAsText(file);
  };

  const isTrained = metricsData?.trained;

  const getPieData = () => {
    if (!batchMutation.data?.summary) return [];
    return [
      { name: 'Positive', value: batchMutation.data.summary.positive, color: '#22c55e' },
      { name: 'Neutral', value: batchMutation.data.summary.neutral, color: '#f59e0b' },
      { name: 'Negative', value: batchMutation.data.summary.negative, color: '#ef4444' },
    ];
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Batch Analysis</h1>
        <p className="text-muted-foreground mt-2">
          Upload datasets to analyze multiple texts simultaneously.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-12">
        <div className="md:col-span-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>Select model and upload data.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
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
              
              <div className="pt-4 space-y-4 border-t">
                <input 
                  type="file" 
                  accept=".csv,.txt" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                />
                <Button 
                  className="w-full" 
                  onClick={() => fileInputRef.current?.click()}
                  disabled={batchMutation.isPending || !isTrained}
                >
                  {batchMutation.isPending ? (
                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</>
                  ) : !isTrained ? (
                    "Models not trained"
                  ) : (
                    <><UploadCloud className="mr-2 h-4 w-4" /> Upload CSV</>
                  )}
                </Button>
                <p className="text-xs text-muted-foreground text-center">
                  CSV format: First column should contain the text
                </p>
              </div>
            </CardContent>
          </Card>

          {batchMutation.data && (
            <Card>
              <CardHeader>
                <CardTitle>Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[200px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={getPieData()}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={2}
                        dataKey="value"
                      >
                        {getPieData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-3 gap-2 mt-4 text-center text-sm">
                  <div>
                    <div className="font-bold text-green-600">{batchMutation.data.summary.positive}</div>
                    <div className="text-muted-foreground text-xs">Positive</div>
                  </div>
                  <div>
                    <div className="font-bold text-amber-600">{batchMutation.data.summary.neutral}</div>
                    <div className="text-muted-foreground text-xs">Neutral</div>
                  </div>
                  <div>
                    <div className="font-bold text-red-600">{batchMutation.data.summary.negative}</div>
                    <div className="text-muted-foreground text-xs">Negative</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="md:col-span-8">
          <Card className="h-full flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle>Results</CardTitle>
                <CardDescription>
                  {batchMutation.data 
                    ? `Analyzed ${batchMutation.data.summary.total} texts in ${batchMutation.data.processingTimeMs}ms` 
                    : "Upload a dataset to see results"}
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent className="flex-1 overflow-auto">
              {batchMutation.data ? (
                <div className="rounded-md border h-[500px] overflow-auto">
                  <Table>
                    <TableHeader className="sticky top-0 bg-background z-10">
                      <TableRow>
                        <TableHead className="w-[60%]">Text</TableHead>
                        <TableHead>Sentiment</TableHead>
                        <TableHead className="text-right">Confidence</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {batchMutation.data.results.map((result, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-medium text-xs max-w-[300px] truncate" title={result.text}>
                            {result.text}
                          </TableCell>
                          <TableCell>
                            <Badge 
                              variant="outline" 
                              className={`text-xs ${
                                result.label === 'positive' ? 'text-green-600 border-green-200 bg-green-50' :
                                result.label === 'negative' ? 'text-red-600 border-red-200 bg-red-50' :
                                'text-amber-600 border-amber-200 bg-amber-50'
                              }`}
                            >
                              {result.label}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground text-xs">
                            {(result.confidence * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="h-full min-h-[400px] flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg">
                  <FileText className="w-12 h-12 opacity-20 mb-4" />
                  <p>Awaiting dataset</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}