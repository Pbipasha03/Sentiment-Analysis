import { useAnalyzeBatch, useGenerateReport, useGetSampleDataset, useGetModelMetrics } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { useState, useEffect } from "react";
import { Loader2, Download, Database } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";

export default function Report() {
  const [format, setFormat] = useState<"json" | "csv">("csv");
  const [includeMetrics, setIncludeMetrics] = useState(true);
  const [datasetSize, setDatasetSize] = useState<number>(100);
  
  const { data: metricsData } = useGetModelMetrics();
  const { data: sampleData, isLoading: loadingSamples } = useGetSampleDataset({ limit: datasetSize });
  const batchMutation = useAnalyzeBatch();
  const reportMutation = useGenerateReport();

  const isTrained = metricsData?.trained;

  const handleGenerate = async () => {
    if (!sampleData?.samples || sampleData.samples.length === 0) {
      toast.error("No data available to analyze");
      return;
    }

    try {
      // 1. Analyze the dataset first
      const texts = sampleData.samples.map(s => s.text);
      const analysisResult = await batchMutation.mutateAsync({ 
        data: { texts, model: "logistic_regression" } 
      });

      // 2. Generate the report based on results
      const reportResult = await reportMutation.mutateAsync({
        data: {
          results: analysisResult.results,
          format: format,
          includeMetrics
        }
      });

      // 3. Trigger download
      const blob = new Blob([reportResult.data], { 
        type: format === 'csv' ? 'text/csv;charset=utf-8;' : 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', reportResult.filename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast.success("Report generated successfully");
    } catch (error) {
      toast.error("Failed to generate report");
      console.error(error);
    }
  };

  const isProcessing = batchMutation.isPending || reportMutation.isPending;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Export Data</h1>
        <p className="text-muted-foreground mt-2">
          Generate comprehensive analysis reports for external academic tools.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 max-w-4xl">
        <Card>
          <CardHeader>
            <CardTitle>Report Configuration</CardTitle>
            <CardDescription>Select format and contents for the export.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="space-y-4">
              <Label className="text-base font-semibold">Output Format</Label>
              <RadioGroup value={format} onValueChange={(v: any) => setFormat(v)} className="grid grid-cols-2 gap-4">
                <div>
                  <RadioGroupItem value="csv" id="csv" className="peer sr-only" />
                  <Label
                    htmlFor="csv"
                    className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                  >
                    <span className="text-lg font-bold uppercase mb-1">CSV</span>
                    <span className="text-xs text-muted-foreground">Spreadsheet format</span>
                  </Label>
                </div>
                <div>
                  <RadioGroupItem value="json" id="json" className="peer sr-only" />
                  <Label
                    htmlFor="json"
                    className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                  >
                    <span className="text-lg font-bold uppercase mb-1">JSON</span>
                    <span className="text-xs text-muted-foreground">Structured data</span>
                  </Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-4 pt-4 border-t">
              <Label className="text-base font-semibold">Dataset Source</Label>
              <div className="flex items-center gap-4 bg-muted/30 p-4 rounded-md">
                <Database className="w-5 h-5 text-muted-foreground" />
                <div className="flex-1">
                  <p className="font-medium text-sm">Sample Validation Set</p>
                  <p className="text-xs text-muted-foreground">
                    {loadingSamples ? "Loading..." : `${sampleData?.samples?.length || 0} texts ready for analysis`}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4 pt-4 border-t">
              <Label className="text-base font-semibold">Inclusions</Label>
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="include-metrics" 
                  checked={includeMetrics} 
                  onCheckedChange={(c: boolean) => setIncludeMetrics(c)} 
                />
                <label
                  htmlFor="include-metrics"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Include Model Performance Metrics block
                </label>
              </div>
            </div>
          </CardContent>
          <CardFooter className="bg-muted/20 py-4 border-t">
            <Button 
              className="w-full" 
              onClick={handleGenerate}
              disabled={isProcessing || !isTrained || loadingSamples}
            >
              {isProcessing ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing Data...</>
              ) : !isTrained ? (
                "Models not trained"
              ) : (
                <><Download className="mr-2 h-4 w-4" /> Download {format.toUpperCase()} Report</>
              )}
            </Button>
          </CardFooter>
        </Card>
        
        <div className="space-y-6">
          <Card className="bg-primary text-primary-foreground border-none">
            <CardHeader>
              <CardTitle className="text-primary-foreground">Data Dictionary</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-4 opacity-90">
              <p>The exported dataset contains the following fields:</p>
              <ul className="list-disc pl-5 space-y-2">
                <li><span className="font-mono bg-primary-foreground/20 px-1 rounded">index</span>: Unique identifier</li>
                <li><span className="font-mono bg-primary-foreground/20 px-1 rounded">text</span>: Original input string</li>
                <li><span className="font-mono bg-primary-foreground/20 px-1 rounded">label</span>: Classification (positive/negative/neutral)</li>
                <li><span className="font-mono bg-primary-foreground/20 px-1 rounded">confidence</span>: Probability score (0.0-1.0)</li>
                <li><span className="font-mono bg-primary-foreground/20 px-1 rounded">model</span>: Architecture used for inference</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}