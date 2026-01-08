
import { ModelOption, ThinkingLevel, AppConfig, ApiProvider } from './types';

export const MODELS: { value: ModelOption; label: string; desc: string; provider: ApiProvider }[] = [
  {
    value: 'gemini-3-flash-preview',
    label: 'Gemini 3 Flash',
    desc: 'Low latency, high throughput, dynamic thinking.',
    provider: 'google'
  },
  {
    value: 'gemini-3-pro-preview',
    label: 'Gemini 3 Pro',
    desc: 'Deep reasoning, complex tasks, higher intelligence.',
    provider: 'google'
  },
  {
    value: 'gpt-4o',
    label: 'GPT-4o',
    desc: 'OpenAI flagship model with vision capabilities.',
    provider: 'openai'
  },
  {
    value: 'gpt-4o-mini',
    label: 'GPT-4o Mini',
    desc: 'Fast, affordable small model for focused tasks.',
    provider: 'openai'
  },
  {
    value: 'o1-preview',
    label: 'O1 Preview',
    desc: 'Advanced reasoning model with chain-of-thought.',
    provider: 'openai'
  },
  {
    value: 'o1-mini',
    label: 'O1 Mini',
    desc: 'Fast reasoning model for efficient problem solving.',
    provider: 'openai'
  },
  {
    value: 'deepseek-chat',
    label: 'DeepSeek Chat',
    desc: 'Advanced language model optimized for conversational AI.',
    provider: 'deepseek'
  },
  {
    value: 'deepseek-coder',
    label: 'DeepSeek Coder',
    desc: 'Specialized model for code generation and programming tasks.',
    provider: 'deepseek'
  },
  {
    value: 'claude-3-5-sonnet-20241022',
    label: 'Claude 3.5 Sonnet',
    desc: 'Anthropic\'s most capable model with excellent reasoning.',
    provider: 'anthropic'
  },
  {
    value: 'claude-3-haiku-20240307',
    label: 'Claude 3 Haiku',
    desc: 'Fast and efficient model for quick responses.',
    provider: 'anthropic'
  },
  {
    value: 'grok-2-1212',
    label: 'Grok 2',
    desc: 'xAI\'s advanced model with real-time knowledge.',
    provider: 'xai'
  },
  {
    value: 'mistral-large-2411',
    label: 'Mistral Large',
    desc: 'Mistral\'s flagship model for complex reasoning.',
    provider: 'mistral'
  },
  {
    value: 'mixtral-8x7b-32768',
    label: 'Mixtral 8x7B',
    desc: 'Mixture of Experts model with excellent performance.',
    provider: 'mistral'
  },
  {
    value: 'custom',
    label: 'Custom Model',
    desc: 'Use any OpenAI-compatible API (LM Studio, Ollama, LocalAI, etc.) by configuring custom base URL.',
    provider: 'custom'
  },
];

export const STORAGE_KEYS = {
  SETTINGS: 'prisma-settings',
  MODEL: 'prisma-selected-model',
  SESSION_ID: 'prisma-active-session-id'
};

export const DEFAULT_CONFIG: AppConfig = {
  planningLevel: 'high',
  expertLevel: 'high',
  synthesisLevel: 'high',
  customApiKey: '',
  customBaseUrl: '',
  enableCustomApi: false,
  enableRecursiveLoop: false,
  apiProvider: 'google',
  customModels: []
};

export const getValidThinkingLevels = (model: ModelOption): ThinkingLevel[] => {
  if (model === 'gemini-3-pro-preview') {
    return ['low', 'high'];
  }
  if (model === 'o1-preview' || model === 'o1-mini') {
    return ['low', 'medium', 'high'];
  }
  return ['minimal', 'low', 'medium', 'high'];
};

export const getThinkingBudget = (level: ThinkingLevel, model: ModelOption): number => {
  const isGeminiPro = model === 'gemini-3-pro-preview';
  const isOpenAIReasoning = model === 'o1-preview' || model === 'o1-mini';

  switch (level) {
    case 'minimal': return 0;
    case 'low': return 2048;
    case 'medium': return 8192;
    case 'high':
      if (isOpenAIReasoning) return 65536;
      if (isGeminiPro) return 32768;
      return 16384;
    default: return 0;
  }
};

export const getProvider = (model: ModelOption): ApiProvider => {
  const modelInfo = MODELS.find(m => m.value === model);
  return modelInfo?.provider || 'google';
};

export const getAllModels = (config: AppConfig): { value: ModelOption; label: string; desc: string; provider: ApiProvider }[] => {
  const presetModels = MODELS.filter(m => m.value !== 'custom');

  const customModels = (config.customModels || []).map(m => ({
    value: m.name as ModelOption,
    label: m.name,
    desc: `Custom ${m.provider} model`,
    provider: m.provider
  }));

  return [...presetModels, ...customModels];
};
