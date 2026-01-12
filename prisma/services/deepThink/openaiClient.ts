
import OpenAI from "openai";
import { ModelOption } from '../../types';
import { withRetry } from '../utils/retry';

export interface OpenAIStreamChunk {
  text: string;
  thought?: string;
}

export interface OpenAIConfig {
  model: ModelOption;
  systemInstruction?: string;
  content: string | Array<any>;
  temperature?: number;
  responseFormat?: 'text' | 'json_object';
  thinkingConfig?: {
    includeThoughts: boolean;
    thinkingBudget: number;
  };
}

const parseThinkingTokens = (text: string): { thought: string; text: string } => {
  const thinkPattern = /<thinking>([\s\S]*?)<\/thinking>/g;
  let thought = '';
  let cleanText = text;

  const matches = text.matchAll(thinkPattern);
  for (const match of matches) {
    thought += match[1];
  }

  cleanText = text.replace(thinkPattern, '');

  return { thought: thought.trim(), text: cleanText.trim() };
};

export const generateContent = async (
  ai: OpenAI,
  config: OpenAIConfig
): Promise<{ text: string; thought?: string }> => {
  const messages: Array<OpenAI.Chat.ChatCompletionMessageParam> = [];

  if (config.systemInstruction) {
    messages.push({
      role: 'system',
      content: config.systemInstruction
    });
  }

  messages.push({
    role: 'user',
    content: config.content as any
  });

  const requestOptions: any = {
    model: config.model,
    messages,
    // Clamp temperature to 1.0 max for compatibility with strict providers (NVIDIA, vLLM, etc.)
    temperature: typeof config.temperature === 'number' ? Math.min(config.temperature, 1.0) : undefined,
  };

  if (config.responseFormat === 'json_object') {
    requestOptions.response_format = { type: 'json_object' };
  }

  try {
    const response = await withRetry(() => ai.chat.completions.create(requestOptions));
    const message = response.choices[0]?.message;
    const content = message?.content || '';
    
    // Check for native reasoning_content field (DeepSeek/NVIDIA style)
    const reasoningContent = (message as any)?.reasoning_content;

    if (reasoningContent && config.thinkingConfig?.includeThoughts) {
       return { text: content, thought: reasoningContent };
    }

    if (config.thinkingConfig?.includeThoughts) {
      const { thought, text } = parseThinkingTokens(content);
      return { text, thought };
    }

    return { text: content };
  } catch (error) {
    console.error('OpenAI generateContent error:', error);
    throw error;
  }
};

export async function* generateContentStream(
  ai: OpenAI,
  config: OpenAIConfig
): AsyncGenerator<OpenAIStreamChunk, void, unknown> {
  const messages: Array<OpenAI.Chat.ChatCompletionMessageParam> = [];

  if (config.systemInstruction) {
    messages.push({
      role: 'system',
      content: config.systemInstruction
    });
  }

  messages.push({
    role: 'user',
    content: config.content as any
  });

  const requestOptions: any = {
    model: config.model,
    messages,
    // Clamp temperature to 1.0 max for compatibility with strict providers
    temperature: typeof config.temperature === 'number' ? Math.min(config.temperature, 1.0) : undefined,
    stream: true,
  };

  const stream = await withRetry(() => ai.chat.completions.create(requestOptions) as any);

  let accumulatedText = '';
  let inThinking = false;
  let currentThought = '';

  for await (const chunk of (stream as any)) {
    const delta = chunk.choices[0]?.delta;
    if (!delta) continue;

    const content = delta.content || '';
    // Check for native reasoning_content field (DeepSeek/NVIDIA style)
    const reasoning = delta.reasoning_content || '';

    // If native reasoning field exists, emit it immediately
    if (reasoning && config.thinkingConfig?.includeThoughts) {
      yield { text: '', thought: reasoning };
    }

    if (content) {
      accumulatedText += content;

      if (config.thinkingConfig?.includeThoughts) {
        // Fallback to tag parsing if reasoning_content wasn't provided but tags exist
        if (content.includes('<thinking>')) {
          inThinking = true;
          continue;
        }

        if (inThinking) {
          if (content.includes('</thinking>')) {
            inThinking = false;
            const parts = content.split('</thinking>', 2);
            currentThought += parts[0];

            if (currentThought.trim()) {
              yield { text: '', thought: currentThought };
              currentThought = '';
            }

            if (parts[1]) {
              yield { text: parts[1], thought: '' };
            }
          } else {
            currentThought += content;
            // Emit thought chunks periodically so it doesn't hang
            if (currentThought.length > 50) {
              yield { text: '', thought: currentThought };
              currentThought = '';
            }
          }
        } else {
          yield { text: content, thought: '' };
        }
      } else {
        yield { text: content, thought: '' };
      }
    }
  }

  if (currentThought.trim()) {
    yield { text: '', thought: currentThought };
  }
}
