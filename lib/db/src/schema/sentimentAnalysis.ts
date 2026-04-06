import { pgTable, text, serial, timestamp, real, integer, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const analysisResultsTable = pgTable("analysis_results", {
  id: serial("id").primaryKey(),
  originalText: text("original_text").notNull(),
  processedText: text("processed_text").notNull(),
  sentimentLabel: text("sentiment_label").notNull(),
  confidence: real("confidence").notNull(),
  modelUsed: text("model_used").notNull(),
  scores: jsonb("scores").notNull(),
  keywords: text("keywords").array().notNull().default([]),
  source: text("source").notNull().default("custom"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertAnalysisResultSchema = createInsertSchema(analysisResultsTable).omit({ id: true, createdAt: true });
export type InsertAnalysisResult = z.infer<typeof insertAnalysisResultSchema>;
export type AnalysisResult = typeof analysisResultsTable.$inferSelect;

export const modelTrainingSessionsTable = pgTable("model_training_sessions", {
  id: serial("id").primaryKey(),
  modelName: text("model_name").notNull(),
  accuracy: real("accuracy").notNull(),
  precision: real("precision_score").notNull(),
  recall: real("recall_score").notNull(),
  f1Score: real("f1_score").notNull(),
  confusionMatrix: jsonb("confusion_matrix").notNull(),
  classReport: jsonb("class_report").notNull(),
  trainingSamples: integer("training_samples").notNull(),
  testSamples: integer("test_samples").notNull(),
  trainingTimeMs: real("training_time_ms").notNull(),
  trainedAt: timestamp("trained_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertModelTrainingSessionSchema = createInsertSchema(modelTrainingSessionsTable).omit({ id: true, trainedAt: true });
export type InsertModelTrainingSession = z.infer<typeof insertModelTrainingSessionSchema>;
export type ModelTrainingSession = typeof modelTrainingSessionsTable.$inferSelect;
