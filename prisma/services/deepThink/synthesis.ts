import { ModelOption, ExpertResult } from '../../types';
import { getSynthesisPrompt } from './prompts';
import { withRetry } from '../utils/retry';
import { generateContentStream as generateOpenAIStream } from './openaiClient';

const isGoogleProvider = (ai: any): boolean => {
  return ai?.models?.generateContentStream !== undefined;
};

export const streamSynthesisResponse = async (
  ai: any,
  model: ModelOption,
  query: string,
  historyContext: string,
  expertResults: ExpertResult[],
  budget: number,
  signal: AbortSignal,
  onChunk: (text: string, thought: string) => void
): Promise<void> => {
  const prompt = getSynthesisPrompt(historyContext, query, expertResults);
  const isGoogle = isGoogleProvider(ai);

  if (isGoogle) {
    const synthesisStream = await withRetry(() => ai.models.generateContentStream({
      model: model,
      contents: prompt,
      config: {
        thinkingConfig: {
          thinkingBudget: budget,
          includeThoughts: true
        }
      }
    }));

    try {
      for await (const chunk of (synthesisStream as any)) {
        if (signal.aborted) break;

        let chunkText = "";
        let chunkThought = "";

        if (chunk.candidates?.[0]?.content?.parts) {
          for (const part of chunk.candidates[0].content.parts) {
            if (part.thought) {
              chunkThought += (part.text || "");
            } else if (part.text) {
              chunkText += part.text;
            }
          }
          onChunk(chunkText, chunkThought);
        }
      }
    } catch (streamError) {
      console.error("Synthesis stream interrupted:", streamError);
      throw streamError;
    }
  } else {
    const stream = generateOpenAIStream(ai, {
      model,
      systemInstruction: undefined,
      content: prompt,
      temperature: 0.7,
      thinkingConfig: {
        thinkingBudget: budget,
        includeThoughts: true
      }
    });

    try {
      for await (const chunk of (stream as any)) {
        if (signal.aborted) break;

        onChunk(chunk.text, chunk.thought || '');
      }
    } catch (streamError) {
      console.error("Synthesis stream interrupted:", streamError);
      throw streamError;
    }
  }
};
