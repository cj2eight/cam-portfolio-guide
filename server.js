// server.js
import 'dotenv/config';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { OpenAI } from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();
const PORT = process.env.PORT || 3000;

// Load embeddings
let knowledgeBase = [];
try {
  const raw = fs.readFileSync('data/portfolio-embeddings.json', 'utf-8');
  knowledgeBase = JSON.parse(raw);
  console.log(`Loaded ${knowledgeBase.length} chunks from knowledge base.`);
} catch (err) {
  console.error(
    'Failed to load data/portfolio-embeddings.json: Did you run npm run index:site?',
    err.message
  );
}

app.use(cors());
app.use(express.json());

// cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0,
    aMag = 0,
    bMag = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aMag += a[i] * a[i];
    bMag += b[i] * b[i];
  }
  return dot / (Math.sqrt(aMag) * Math.sqrt(bMag) + 1e-8);
}

async function getRelevantContext(query, topK = 5) {
  if (!knowledgeBase.length) return '';

  const embeddingRes = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: query
  });

  const queryEmbedding = embeddingRes.data[0].embedding;

  const scored = knowledgeBase.map((item) => ({
    ...item,
    score: cosineSimilarity(queryEmbedding, item.embedding)
  }));

  scored.sort((a, b) => b.score - a.score);

  const top = scored.slice(0, topK);

  const contextText = top
    .map(
      (item, idx) =>
        `Source ${idx + 1} (${item.url}):\n${item.content}`
    )
    .join('\n\n---\n\n');

  return contextText;
}

// Main route for chatbot
app.post('/api/portfolio-guide', async (req, res) => {
  try {
    const { message, history = [] } = req.body || {};
    if (!message) {
      return res.status(400).json({ error: 'Missing message' });
    }

    const context = await getRelevantContext(message);

    const systemPrompt = `
You are "Cam's portfolio guide", an AI assistant that answers questions about Cam Merriwether and his design work.

Only use the provided context from his website to answer. If you don't see the answer in the context, say you're not sure and guide the user where they can look on the site.

Be clear, friendly, and concise.
`;

    const messages = [
      { role: 'system', content: systemPrompt },
      {
        role: 'system',
        content: `Website context:\n\n${context}`
      }
    ];

    // Add previous turns
    for (const turn of history.slice(-6)) {
      messages.push({ role: 'user', content: turn.user });
      messages.push({ role: 'assistant', content: turn.assistant });
    }

    messages.push({ role: 'user', content: message });

    const completion = await client.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages,
      temperature: 0.4
    });

    const reply = completion.choices[0].message.content;
    res.json({ reply });
  } catch (err) {
    console.error('Error in /api/portfolio-guide:', err);
    res.status(500).json({
      error: 'Server error',
      details: err.message
    });
  }
});

app.get('/', (req, res) => {
  res.send('Cam portfolio guide API is running.');
});

app.listen(PORT, () => {
  console.log(`Cam portfolio guide API is listening on port ${PORT}`);
});