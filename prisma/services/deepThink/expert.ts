import { ModelOption, ExpertResult } from '../../types';
import { getExpertSystemInstruction } from './prompts';
import { withRetry } from '../utils/retry';
import { generateContentStream as generateOpenAIStream } from './openaiClient';

const isGoogleProvider = (ai: any): boolean => {
  return ai?.models?.generateContentStream !== undefined;
};

export const streamExpertResponse = async (
  ai: any,
  model: ModelOption,
  expert: ExpertResult,
  context: string,
  budget: number,
  signal: AbortSignal,
  onChunk: (text: string, thought: string) => void
): Promise<void> => {
  const isGoogle = isGoogleProvider(ai);

  if (isGoogle) {
    const streamResult = await withRetry(() => ai.models.generateContentStream({
      model: model,
      contents: expert.prompt,
      config: {
        systemInstruction: getExpertSystemInstruction(expert.role, expert.description, context),
        temperature: expert.temperature,
        thinkingConfig: {
          thinkingBudget: budget,
          includeThoughts: true
        }
      }
    }));

    try {
      for await (const chunk of (streamResult as any)) {
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
      console.error(`Stream interrupted for expert ${expert.role}:`, streamError);
      throw streamError;
    }
  } else {
    const stream = generateOpenAIStream(ai, {
      model,
      systemInstruction: getExpertSystemInstruction(expert.role, expert.description, context),
      content: expert.prompt,
      temperature: expert.temperature,
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
      console.error(`Stream interrupted for expert ${expert.role}:`, streamError);
      throw streamError;
    }
  }
};
