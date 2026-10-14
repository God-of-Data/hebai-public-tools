// One-call access check for the JEV_API_KEY secret: tries TypeSafe directly, then OpenRouter's Decisions API.
// Sends one short Hebrew sentence written for this test (no data). Prints only status, model, token usage, errors.
import { TypeSafeClient } from '@typesafe-ai/sdk';
const key = (process.env.JEV_API_KEY || '').trim();
const text = 'מחר בבוקר נצא לטיול בצפון, כדאי להביא מים, כובע ונעליים נוחות. נפגשים בשמונה ליד בית הספר.';
const Q = { formal: { type: 'noul', instructions: 'The text is a formal institutional notice.' } };
const status = { key_present: !!key, attempts: [] };
for (const [name, opts, model] of [['typesafe-latest', { apiKey: key, timeout: 30000 }, 'jev-latest'], ['typesafe-default', { apiKey: key, timeout: 30000 }, undefined],
                                   ['openrouter', { apiKey: key, baseURL: 'https://openrouter.ai/api', timeout: 30000 }, 'typesafe/jev-1.13'],
                                   ['openrouter-short', { apiKey: key, baseURL: 'https://openrouter.ai/api', timeout: 30000 }, 'jev-1.13']]) {
  try {
    const res = await new TypeSafeClient(opts).systemOne(model ? { model, state: text, questions: Q } : { state: text, questions: Q });
    status.attempts.push({ name, ok: true, model: res.model, usage: res.usage, answer: res.answers?.formal });
    break;
  } catch (e) { status.attempts.push({ name, ok: false, error: String(e).slice(0, 300) }); }
}
console.log('::notice::' + JSON.stringify(status).slice(0, 1500));
import fs from 'node:fs'; fs.mkdirSync('out', { recursive: true }); fs.writeFileSync('out/status.json', JSON.stringify(status, null, 1));
