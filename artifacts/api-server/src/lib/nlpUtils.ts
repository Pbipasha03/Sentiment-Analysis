import type { SentimentLabel } from "./mlModels.js";

// Negation words must be preserved for sentiment detection.
const NEGATION_WORDS = new Set(["not", "no", "never", "neither", "nor", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mightn't"]);

const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "could", "should", "may", "might", "shall", "can", "this", "that",
  "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
  "your", "he", "she", "they", "them", "his", "her", "their", "what",
  "which", "who", "when", "where", "how", "so", "if",
  "about", "up", "out", "as", "just", "also", "than", "more", "all",
  "am", "been", "before", "after", "too", "then", "now", "here",
  "there", "some", "any", "each", "every", "both", "few", "most",
  "other", "into", "through", "during", "until", "while", "although",
  "because", "since", "though", "however", "therefore", "rt", "http",
  "https", "www", "com", "co", "de", "en", "el", "la", "le", "les"
]);

// Negation patterns - words/phrases that should be converted to their opposite sentiment.
const NEGATION_DICT: Record<string, string> = {
  "neither good nor bad": "neutral",
  "not bad but not great": "okay",
  "not too good not too bad": "neutral",
  "not too bad": "okay",
  "not bad": "okay",
  "not satisfied": "unsatisfied",
  "not good": "bad",
  "not great": "bad",
  "not excellent": "terrible",
  "not amazing": "terrible",
  "not happy": "sad",
  "not interested": "disinterested",
  "not impressive": "disappointing",
  "not impressed": "disappointed",
  "not acceptable": "unacceptable",
  "not helpful": "unhelpful",
  "not useful": "useless",
  "not work": "broken",
  "does not work": "broken",
  "doesn't work": "broken",
  "do not work": "broken",
  "don't work": "broken",
  "not understand": "confused",
  "not sure": "uncertain",
  "not recommend": "unrecommendable",
  "do not recommend": "unrecommendable",
  "don't recommend": "unrecommendable",
  "cannot recommend": "unrecommendable",
  "can't recommend": "unrecommendable",
  "not worth": "worthless",
  "don't like": "dislike",
  "don't want": "unwanted",
  "never liked": "hated",
  "no good": "bad"
};

const STRONG_POSITIVE_WORDS = new Set([
  "love", "amazing", "excellent", "fantastic", "wonderful", "perfect",
  "best", "good", "great", "awesome", "happy", "satisfied", "outstanding", "thrilled", "grateful",
  "beautiful", "incredible", "efficient", "excited", "proud", "recommend",
  "impressed", "superb", "phenomenal", "divine", "breathtaking", "overjoyed",
  "blessed", "masterpiece", "won", "accepted", "promoted", "like", "liked",
  "helpful", "useful", "reliable", "pleasant", "smooth", "easy",
]);

const STRONG_NEGATIVE_WORDS = new Set([
  "hate", "terrible", "awful", "horrible", "worst", "disgusting", "broken",
  "useless", "poor", "disappointed", "disappointing", "unsatisfied",
  "unacceptable", "worthless", "bad", "sad", "unhappy", "frustrated",
  "furious", "miserable", "scam", "scammed", "fraudulent", "refund", "waste",
  "regret", "regretting", "unreliable", "rude", "toxic", "failed", "failure",
  "disaster", "drained", "exhausted", "cancelled", "ignored", "complaint",
  "cold", "wrong", "unhelpful", "dislike", "unrecommendable",
]);

const NEUTRAL_PHRASES = [
  "neither good nor bad",
  "not bad but not great",
  "not bad but not excellent",
  "not too good not too bad",
  "not too bad",
  "not bad",
  "okay",
  "average",
  "mediocre",
  "so so",
  "standard",
  "normal",
];

export interface RuleBasedSentiment {
  label: SentimentLabel;
  confidence: number;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Handle negation patterns BEFORE other preprocessing
function handleNegation(text: string): string {
  let processed = text.toLowerCase();
  // Replace negation patterns with their opposite sentiments
  for (const [pattern, replacement] of Object.entries(NEGATION_DICT)) {
    processed = processed.replace(new RegExp(`\\b${escapeRegExp(pattern)}\\b`, "g"), replacement);
  }
  return processed;
}

export function preprocessText(text: string): string {
  // Step 1: Handle negation FIRST
  let processed = handleNegation(text);
  
  // Step 2: Clean URLs and mentions
  processed = processed.replace(/https?:\/\/\S+/g, "");
  processed = processed.replace(/@\w+/g, "");
  processed = processed.replace(/#(\w+)/g, "$1");
  
  // Step 3: Clean special characters
  processed = processed.replace(/[^a-z0-9\s']/g, " ");
  processed = processed.replace(/\s+/g, " ").trim();
  return processed;
}

export function tokenize(text: string): string[] {
  const processed = preprocessText(text);
  // Keep negation words (they contain negative sentiment)
  return processed.split(" ").filter(w => w.length > 2 && (!STOP_WORDS.has(w) || NEGATION_WORDS.has(w)));
}

export function classifyWithRules(text: string): RuleBasedSentiment | null {
  const lower = text.toLowerCase();

  for (const phrase of NEUTRAL_PHRASES) {
    if (new RegExp(`\\b${escapeRegExp(phrase)}\\b`).test(lower)) {
      return { label: "neutral", confidence: 0.9 };
    }
  }

  const tokens = tokenize(text);
  let positive = 0;
  let negative = 0;

  for (const token of tokens) {
    if (STRONG_POSITIVE_WORDS.has(token)) positive += 1;
    if (STRONG_NEGATIVE_WORDS.has(token)) negative += 1;
  }

  if (positive === 0 && negative === 0) return null;
  if (negative > positive) return { label: "negative", confidence: negative >= 2 ? 0.95 : 0.9 };
  if (positive > negative) return { label: "positive", confidence: positive >= 2 ? 0.95 : 0.9 };

  return { label: "neutral", confidence: 0.75 };
}

export function extractKeywords(text: string, topN: number = 10): string[] {
  const tokens = tokenize(text);
  const freq: Record<string, number> = {};
  for (const token of tokens) {
    freq[token] = (freq[token] || 0) + 1;
  }
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([word]) => word);
}

export function getWordFrequencies(texts: string[]): Array<{ text: string; value: number }> {
  const freq: Record<string, number> = {};
  for (const text of texts) {
    const tokens = tokenize(text);
    for (const token of tokens) {
      freq[token] = (freq[token] || 0) + 1;
    }
  }
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 100)
    .map(([word, count]) => ({ text: word, value: count }));
}

export function createBigrams(tokens: string[]): string[] {
  const bigrams: string[] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    bigrams.push(`${tokens[i]}_${tokens[i + 1]}`);
  }
  return bigrams;
}

export function buildVocabulary(texts: string[]): string[] {
  const vocab = new Set<string>();
  for (const text of texts) {
    const tokens = tokenize(text);
    const bigrams = createBigrams(tokens);
    tokens.forEach(t => vocab.add(t));
    bigrams.forEach(b => vocab.add(b));
  }
  return Array.from(vocab).sort();
}

export function textToTfIdf(text: string, vocab: string[], idf: Record<string, number>): number[] {
  const tokens = tokenize(text);
  const bigrams = createBigrams(tokens);
  const allTokens = [...tokens, ...bigrams];
  const tf: Record<string, number> = {};
  const total = allTokens.length || 1;
  for (const t of allTokens) {
    tf[t] = (tf[t] || 0) + 1;
  }
  return vocab.map(v => {
    const tfVal = (tf[v] || 0) / total;
    const idfVal = idf[v] || 0;
    return tfVal * idfVal;
  });
}

export function computeIdf(texts: string[], vocab: string[]): Record<string, number> {
  const N = texts.length;
  const idf: Record<string, number> = {};
  for (const v of vocab) {
    let docCount = 0;
    for (const text of texts) {
      const tokens = tokenize(text);
      const bigrams = createBigrams(tokens);
      if (tokens.includes(v) || bigrams.includes(v)) {
        docCount++;
      }
    }
    idf[v] = Math.log((N + 1) / (docCount + 1)) + 1;
  }
  return idf;
}

export function computeBagOfWords(text: string, vocab: string[]): number[] {
  const tokens = tokenize(text);
  const bigrams = createBigrams(tokens);
  const allTokens = new Set([...tokens, ...bigrams]);
  return vocab.map(v => (allTokens.has(v) ? 1 : 0));
}
