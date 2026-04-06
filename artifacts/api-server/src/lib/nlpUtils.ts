const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "could", "should", "may", "might", "shall", "can", "this", "that",
  "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
  "your", "he", "she", "they", "them", "his", "her", "their", "what",
  "which", "who", "when", "where", "how", "not", "no", "so", "if",
  "about", "up", "out", "as", "just", "also", "than", "more", "all",
  "am", "been", "before", "after", "very", "too", "then", "now", "here",
  "there", "some", "any", "each", "every", "both", "few", "more", "most",
  "other", "into", "through", "during", "until", "while", "although",
  "because", "since", "though", "however", "therefore", "rt", "http",
  "https", "www", "com", "co", "de", "en", "el", "la", "le", "les"
]);

export function preprocessText(text: string): string {
  let processed = text.toLowerCase();
  processed = processed.replace(/https?:\/\/\S+/g, "");
  processed = processed.replace(/@\w+/g, "");
  processed = processed.replace(/#(\w+)/g, "$1");
  processed = processed.replace(/[^a-z0-9\s']/g, " ");
  processed = processed.replace(/\s+/g, " ").trim();
  return processed;
}

export function tokenize(text: string): string[] {
  const processed = preprocessText(text);
  return processed.split(" ").filter(w => w.length > 2 && !STOP_WORDS.has(w));
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
