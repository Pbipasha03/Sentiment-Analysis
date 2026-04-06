import { useGetModelMetrics, useHealthCheck } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, BrainCircuit, CheckCircle2, AlertCircle } from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";

export default function Home() {
  const { data: health, isLoading: healthLoading } = useHealthCheck();
  const { data: metricsData, isLoading: metricsLoading } = useGetModelMetrics();

  const isTrained = metricsData?.trained;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Microtext Sentiment Analysis platform overview and system status.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold flex items-center gap-2">
              {healthLoading ? (
                <span className="text-muted-foreground text-sm">Checking...</span>
              ) : health?.status === "ok" ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="text-green-500">Operational</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 text-red-500" />
                  <span className="text-red-500">Unavailable</span>
                </>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Backend NLP Service</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            <BrainCircuit className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold flex items-center gap-2">
              {metricsLoading ? (
                <span className="text-muted-foreground text-sm">Loading...</span>
              ) : isTrained ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  <span>Trained</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 text-amber-500" />
                  <span className="text-amber-500">Untrained</span>
                </>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {isTrained ? "Models are ready for inference" : "Training required before use"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quick Analysis</CardTitle>
            <CardDescription>
              Test the sentiment analysis model on a single text snippet.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full">
              <Link href="/analyze">Go to Analyzer</Link>
            </Button>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Model Management</CardTitle>
            <CardDescription>
              View performance metrics, confusion matrices, and retrain models.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline" className="w-full">
              <Link href="/models">View Models</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}