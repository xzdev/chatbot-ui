import { NextApiRequest, NextApiResponse } from 'next';

import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

import { Tiktoken } from '@dqbd/tiktoken';
import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';

export const config = {
  runtime: 'nodejs',
};

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  try {
    const { model, messages, key, prompt, temperature } = req.body as ChatBody;

    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const tokens = encoding.encode(message.content);

      if (tokenCount + tokens.length + 1000 > model.tokenLimit) {
        break;
      }
      tokenCount += tokens.length;
      messagesToSend = [message, ...messagesToSend];
    }

    encoding.free();

    return await OpenAIStream(
      {
        model,
        systemPrompt: promptToSend,
        temperature: temperatureToUse,
        key,
        messages: messagesToSend,
      },
      res,
    );
  } catch (error) {
    console.error(error);
    return res.status(500).json(error);
  }
};

export default handler;
