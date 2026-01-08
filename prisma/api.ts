import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";
import { ApiProvider, AppConfig, CustomModel } from './types';

type AIProviderConfig = {
  provider?: ApiProvider;
  apiKey?: string;
  baseUrl?: string;
};

/**
 * Find custom model configuration by model name
 */
export const findCustomModel = (modelName: string, customModels?: CustomModel[]): CustomModel | undefined => {
  return customModels?.find(m => m.name === modelName);
};

export const getAI = (config?: AIProviderConfig) => {
  const provider = config?.provider || 'google';
  // Support both Vite env vars (VITE_) and standard env vars for flexibility
  const apiKey = config?.apiKey || (import.meta.env as any).VITE_API_KEY || process.env.API_KEY;

  if (provider === 'openai' || provider === 'deepseek' || provider === 'custom' || provider === 'anthropic' || provider === 'xai' || provider === 'mistral') {
    const options: any = {
      apiKey: apiKey,
      // WARNING: dangerouslyAllowBrowser enables client-side API calls
      // This is acceptable for local development but NOT production
      // In production, use a backend proxy to protect API keys
      dangerouslyAllowBrowser: true,
    };

    if (config?.baseUrl) {
      options.baseURL = config.baseUrl;
    } else if (provider === 'deepseek') {
      options.baseURL = 'https://api.deepseek.com/v1';
    } else if (provider === 'anthropic') {
      options.baseURL = 'https://api.anthropic.com/v1';
    } else if (provider === 'xai') {
      options.baseURL = 'https://api.x.ai/v1';
    } else if (provider === 'mistral') {
      options.baseURL = 'https://api.mistral.ai/v1';
    }

    return new OpenAI(options);
  } else {
    const options: any = {
      apiKey: apiKey,
    };

    if (config?.baseUrl) {
      options.baseUrl = config.baseUrl;
    }

    return new GoogleGenAI(options);
  }
};

export const getAIProvider = (model: string): ApiProvider => {
  if (model.startsWith('gpt-') || model.startsWith('o1-')) {
    return 'openai';
  }
  if (model.startsWith('deepseek-')) {
    return 'deepseek';
  }
  if (model.startsWith('claude-')) {
    return 'anthropic';
  }
  if (model.startsWith('grok-')) {
    return 'xai';
  }
  if (model.startsWith('mistral-') || model.startsWith('mixtral-')) {
    return 'mistral';
  }
  if (model === 'custom') {
    return 'custom';
  }
  return 'google';
};