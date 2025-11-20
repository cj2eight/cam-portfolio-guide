// server.js
// server.js
import 'dotenv/config';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { OpenAI } from 'openai';

// Crash logging so Render shows hidden errors
process.on('unhandledRejection', (err) => {
  console.error('UNHANDLED REJECTION', err);
});
process.on('uncaughtException', (err) => {
  console.error('UNCAUGHT EXCEPTION', err);
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// ---- Load embeddings from file ----
let knowledgeBase = [];
try {
  const raw = fs.readFileSync('data/portfolio-embeddings.json', 'utf8');
  knowledgeBase = JSON.parse(raw);
  console.log(`Loaded ${knowledgeBase.length} chunks from knowledge base.`);
} catch (err) {
  console.error('ERROR: Could not load embeddings file:', err);
}

// ---- Cosine similarity helper ----
function cosineSimilarity(a, b) {
  let dot = 0;
  let aMag = 0;
  let bMag = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aMag += a[i] * a[i];
    bMag += b[i] * b[i];
  }
  return dot / (Math.sqrt(aMag) * Math.sqrt(bMag) + 1e-8);
}

// ---- Retrieve relevant context for a query ----
async function getRelevantContext(query, topK = 6) {
  if (!knowledgeBase.length) return '';

  const embed = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: query
  });

  const queryEmbedding = embed.data[0].embedding;

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

// ---- Health check route ----
app.get('/', (req, res) => {
  res.send('Cam portfolio guide API is running.');
});

// ---- Main chat route ----
app.post('/api/portfolio-guide', async (req, res) => {
  try {
    const { message = '', history = [] } = req.body || {};

    const context = await getRelevantContext(message);

    const systemPrompt = `
You are "Cam's portfolio guide", an AI assistant that answers questions ONLY about product designer Cameron Merriwether and his work.

Assume that "Cam", "Cameron", or "he" ALWAYS refer to this same person: the owner of camjmerriwether.com. Do NOT ask which Cam the user means.

You must base your answers ONLY on the website context provided. If the answer is not clearly supported by the context, say you are not sure and suggest where the user can look on the site (for example: specific case studies, About page, or resume).

Be concise, specific, and focused on his product design experience, skills, industries (like fintech, sports, etc.), and project outcomes.
`;

    const messages = [
      { role: 'system', content: systemPrompt },
      {
        role: 'system',
        content: `Website context from Cam's portfolio:\n\n${context}`
      }
    ];

    // include short recent history for continuity
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
  } catch (error) {
    console.error('ERROR in /api/portfolio-guide:', error);
    res.status(500).json({
      reply: "Sorry, the portfolio guide ran into a server error. Try again in a bit."
    });
  }
});

// ---- Start server ----
app.listen(PORT, () => {
  console.log(`Cam portfolio guide API is listening on port ${PORT}`);
});