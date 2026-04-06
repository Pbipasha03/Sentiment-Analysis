import { useGetWordCloudData, useGetSampleDataset } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState, useEffect } from "react";
import { Loader2, Cloud } from "lucide-react";

export default function WordCloud() {
  const [filter, setFilter] = useState<string>("all");
  
  // We'll use the sample dataset to generate the word cloud
  const { data: sampleData, isLoading: isSamplesLoading } = useGetSampleDataset({ limit: 1000 });
  const cloudMutation = useGetWordCloudData();

  useEffect(() => {
    if (sampleData?.samples) {
      const texts = sampleData.samples.map(s => s.text);
      const sentimentFilter = filter === "all" ? undefined : filter as any;
      cloudMutation.mutate({ data: { texts, sentimentFilter } });
    }
  }, [sampleData, filter]);

  // Determine colors based on filter
  const getWordColor = (value: number, max: number) => {
    const intensity = Math.max(0.4, value / max);
    
    if (filter === 'positive') return `rgba(34, 197, 94, ${intensity + 0.2})`;
    if (filter === 'negative') return `rgba(239, 68, 68, ${intensity + 0.2})`;
    if (filter === 'neutral') return `rgba(245, 158, 11, ${intensity + 0.2})`;
    
    // Mixed default colors
    return `hsl(var(--primary) / ${intensity + 0.2})`;
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Lexical Analysis</h1>
          <p className="text-muted-foreground mt-2">
            Visualize term frequency distribution across sentiment classes.
          </p>
        </div>
        <div className="w-[200px]">
          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger>
              <SelectValue placeholder="Filter by Sentiment" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sentiments</SelectItem>
              <SelectItem value="positive">Positive Only</SelectItem>
              <SelectItem value="neutral">Neutral Only</SelectItem>
              <SelectItem value="negative">Negative Only</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="min-h-[500px] flex flex-col">
        <CardHeader>
          <CardTitle>N-Gram Frequency Cloud</CardTitle>
          <CardDescription>Size indicates relative frequency in the analyzed corpus.</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center p-8 bg-muted/10">
          {(isSamplesLoading || cloudMutation.isPending) ? (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <Loader2 className="h-12 w-12 animate-spin mb-4" />
              <p>Generating visualization...</p>
            </div>
          ) : cloudMutation.data?.words && cloudMutation.data.words.length > 0 ? (
            <div className="flex flex-wrap justify-center content-center gap-4 max-w-4xl h-full leading-none">
              {(() => {
                const words = cloudMutation.data.words;
                const maxVal = Math.max(...words.map(w => w.value));
                const minVal = Math.min(...words.map(w => w.value));
                
                return words.map((word, i) => {
                  // Normalize size between 14px and 64px
                  const normSize = 14 + ((word.value - minVal) / (maxVal - minVal || 1)) * 50;
                  
                  return (
                    <span 
                      key={i}
                      className="font-bold inline-block transition-transform hover:scale-110 cursor-default"
                      style={{ 
                        fontSize: `${normSize}px`,
                        color: getWordColor(word.value, maxVal),
                        opacity: 0.8 + (word.value / maxVal) * 0.2
                      }}
                      title={`Frequency: ${word.value}`}
                    >
                      {word.text}
                    </span>
                  );
                });
              })()}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <Cloud className="h-16 w-16 mb-4 opacity-20" />
              <p>No word data available</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}