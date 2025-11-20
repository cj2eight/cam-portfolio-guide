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

// Create OpenAI client once
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Create express app once
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Load embeddings
let knowledgeBase;
try {
  const raw = fs.readFileSync('data/portfolio-embeddings.json', 'utf8');
  knowledgeBase = JSON.parse(raw);
  console.log(`Loaded ${knowledgeBase.length} chunks from knowledge base.`);
} catch (err) {
  console.error("ERROR: Could not load embeddings file:", err);
}

// Root route for health check
app.get('/', (req, res) => {
  res.send('Cam portfolio guide API is running.');
});

// Chat route
app.post('/api/portfolio-guide', async (req, res) => {
  try {
    const { message = '', history = [] } = req.body;

    // Build messages array
    const messages = [
      ...history.flatMap((pair) => [
        { role: 'user', content: pair.user },
        { role: 'assistant', content: pair.assistant }
      ]),
      { role: 'user', content: message }
    ];

    // Call OpenAI (adjust for your model or config)
    const completion = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages
    });

    res.json({ reply: completion.choices[0].message.content });

  } catch (error) {
    console.error('ERROR in /api/portfolio-guide:', error);
    res.status(500).json({ reply: "Server error. Try again later." });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Cam portfolio guide API is listening on port ${PORT}`);
});