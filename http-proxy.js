const http = require('http');
const https = require('https');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');

// Keep-alive agent: reuses TLS connections to opencode.ai across requests
const upstreamAgent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 16,
  maxFreeSockets: 8,
});

const TARGET_HOST = 'opencode.ai';
const TARGET_API_PATH = '/zen/go/v1';
const TARGET_API_KEY = process.env.OPENCODE_API_KEY || process.env.ZAI_API_KEY;
const LOG_FILE = process.env.PROXY_LOG_FILE || path.join(__dirname, 'requests.log');

if (!TARGET_API_KEY) {
  console.error('Missing OPENCODE_API_KEY or ZAI_API_KEY environment variable.');
  process.exit(1);
}

const GLOBAL_PENDING_TOOL_CALLS = new Map();
const MAX_PENDING_TOOL_CALLS = 1000;

const MODEL_MAP = {
  // OpenAI -> GLM defaults
  'gpt-5.5': 'glm-5.1',
  'gpt-5.4': 'glm-5',
  'gpt-5': 'glm-5.1',
  'gpt-4o': 'glm-4.7',
  'gpt-4': 'glm-4.7',
  'gpt-4o-mini': 'glm-4.5-air',
  'o3-mini': 'glm-4.7',
  'o1': 'glm-5.1',
  'gpt-5.4-mini': 'glm-5',
  'gpt-5.1': 'glm-5.1',
  'gpt-5.3': 'glm-5',
  'gpt-5.3-mini': 'glm-5',
  // Identity passthroughs - allow direct model selection in config.toml
  'deepseek-v4-pro': 'deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-v4-flash',
  'minimax-m2.7': 'minimax-m2.7',
  'minimax-m2.5': 'minimax-m2.5',
  'kimi-k2.6': 'kimi-k2.6',
  'glm-5.1': 'glm-5.1',
  'glm-5': 'glm-5',
  'qwen3.6-plus': 'qwen3.6-plus',
};

function log(...args) {
  const line = `[${new Date().toISOString()}] ${args.join(' ')}`;
  console.log(line);
  try { fs.appendFileSync(LOG_FILE, line + '\n'); } catch (e) {}
}

function decompressBody(buffer, encoding) {
  if (!buffer || buffer.length === 0) return buffer;
  // Auto-detect by magic bytes if encoding header is missing/wrong
  if (!encoding && buffer.length >= 4) {
    const b = buffer;
    if (b[0] === 0x1f && b[1] === 0x8b) encoding = 'gzip';
    else if (b[0] === 0x28 && b[1] === 0xb5 && b[2] === 0x2f && b[3] === 0xfd) encoding = 'zstd';
  }
  try {
    if (encoding === 'gzip' || encoding === 'x-gzip') return zlib.gunzipSync(buffer);
    if (encoding === 'deflate') return zlib.inflateSync(buffer);
    if (encoding === 'br') return zlib.brotliDecompressSync(buffer);
    if (encoding === 'zstd' && zlib.zstdDecompressSync) return zlib.zstdDecompressSync(buffer);
  } catch (e) {
    log('Decompress error (' + encoding + '):', e.message);
  }
  return buffer;
}

function normalizeRole(role) {
  // Chat Completions doesn't accept "developer" - map to "system"
  if (role === 'developer') return 'system';
  return role || 'user';
}

function flattenContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return content || '';
  return content.map(c => {
    if (typeof c === 'string') return c;
    if (c.type === 'input_text' && c.text) return c.text;
    if (c.type === 'output_text' && c.text) return c.text;
    if (c.text) return c.text;
    return '';
  }).filter(Boolean).join('\n');
}

function translateTools(tools) {
  if (!Array.isArray(tools)) return tools;
  const out = [];
  for (const t of tools) {
    // Already Chat Completions format
    if (t.function) { out.push(t); continue; }
    // Responses API function tool
    if (t.type === 'function' && t.name) {
      out.push({
        type: 'function',
        function: {
          name: t.name,
          description: t.description || '',
          parameters: t.parameters || { type: 'object', properties: {} },
        },
      });
      continue;
    }
    // Custom tool type from Codex
    if (t.type === 'custom' && t.name) {
      out.push({
        type: 'function',
        function: {
          name: t.name,
          description: t.description || '',
          parameters: t.input_schema || t.parameters || { type: 'object', properties: {} },
        },
      });
      continue;
    }
    // Responses namespace tools (for example MCP tools) are flattened for Chat Completions.
    if (t.type === 'namespace' && t.name && Array.isArray(t.tools)) {
      const namePrefix = t.name.endsWith('__') ? t.name : `${t.name}__`;
      for (const subTool of t.tools) {
        if (subTool.type !== 'function' || !subTool.name) continue;
        out.push({
          type: 'function',
          function: {
            name: `${namePrefix}${subTool.name}`,
            description: subTool.description || `${t.name}.${subTool.name}`,
            parameters: subTool.parameters || { type: 'object', properties: {} },
          },
        });
      }
      continue;
    }
    // Skip unsupported tool types (web_search, image_generation, namespace, code_interpreter, etc.)
    log(`Skipping unsupported tool type: ${t.type} (${t.name || 'unnamed'})`);
  }
  return out;
}

function splitMcpToolName(name) {
  if (typeof name !== 'string' || !name.startsWith('mcp__')) return null;
  const separator = name.indexOf('__', 'mcp__'.length);
  if (separator < 0) return null;
  const namespace = name.slice(0, separator + 2);
  const toolName = name.slice(separator + 2);
  if (!toolName) return null;
  return { namespace, name: toolName };
}

function upstreamToolNameFromCodexItem(item) {
  if (item?.namespace && item?.name) {
    const namespace = item.namespace.endsWith('__') ? item.namespace : `${item.namespace}__`;
    return `${namespace}${item.name}`;
  }
  return item?.name;
}

function codexToolCallItem(t, argumentsText, status = 'completed') {
  const split = splitMcpToolName(t.name);
  const item = {
    id: t.itemId,
    type: 'function_call',
    status,
    call_id: t.id,
    name: split ? split.name : t.name,
    arguments: argumentsText,
  };
  if (split) item.namespace = split.namespace;
  return item;
}

function codexToolDisplayName(name) {
  const split = splitMcpToolName(name);
  return split ? `${split.namespace}${split.name}` : name;
}

const MAX_TOOL_OUTPUT_CHARS = 60000;
const MAX_TOOL_TEXT_PART_CHARS = 25000;

function truncateText(text, maxChars = MAX_TOOL_OUTPUT_CHARS) {
  if (typeof text !== 'string' || text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n[truncated ${text.length - maxChars} chars]`;
}

function compactToolOutputParts(parts) {
  if (!Array.isArray(parts)) return null;
  const lines = [];
  for (const part of parts) {
    if (typeof part === 'string') {
      lines.push(truncateText(part, MAX_TOOL_TEXT_PART_CHARS));
      continue;
    }
    if (!part || typeof part !== 'object') continue;
    if ((part.type === 'input_text' || part.type === 'text') && typeof part.text === 'string') {
      lines.push(truncateText(part.text, MAX_TOOL_TEXT_PART_CHARS));
      continue;
    }
    if (typeof part.text === 'string') {
      lines.push(truncateText(part.text, MAX_TOOL_TEXT_PART_CHARS));
      continue;
    }
    if (part.type === 'input_image' || part.image_url || part.image_url?.url) {
      const image = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url || '';
      lines.push(`[screenshot omitted: ${image.length} chars]`);
      continue;
    }
    lines.push(truncateText(JSON.stringify(part), 2000));
  }
  return truncateText(lines.filter(Boolean).join('\n'));
}

function stringifyToolOutput(output) {
  if (typeof output === 'string') {
    const trimmed = output.trim();
    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) {
          const compacted = compactToolOutputParts(parsed);
          if (compacted) return compacted;
        }
        if (Array.isArray(parsed?.content)) {
          const compacted = compactToolOutputParts(parsed.content);
          if (compacted) return compacted;
        }
      } catch (e) {}
    }
    return truncateText(output);
  }
  if (Array.isArray(output)) {
    const compacted = compactToolOutputParts(output);
    if (compacted) return compacted;
  }
  if (Array.isArray(output?.content)) {
    const compacted = compactToolOutputParts(output.content);
    if (compacted) return compacted;
  }
  return truncateText(JSON.stringify(output || ''));
}

function estimateTokens(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value || '');
  return Math.max(1, Math.ceil(text.length / 4));
}

function normalizeUsage(upstreamUsage, requestData, outputText, toolCalls = {}) {
  const inputTokens = upstreamUsage?.prompt_tokens ?? upstreamUsage?.input_tokens;
  const outputTokens = upstreamUsage?.completion_tokens ?? upstreamUsage?.output_tokens;
  const toolOutputText = Object.values(toolCalls).map(t => `${t.name || ''} ${t.args || ''}`).join('\n');
  const fallbackInput = estimateTokens(requestData?.messages || requestData?.input || '');
  const fallbackOutput = estimateTokens(`${outputText || ''}\n${toolOutputText}`);
  const input = Number.isFinite(inputTokens) && inputTokens > 0 ? inputTokens : fallbackInput;
  const output = Number.isFinite(outputTokens) && outputTokens > 0 ? outputTokens : fallbackOutput;
  const totalTokens = upstreamUsage?.total_tokens;
  const total = Number.isFinite(totalTokens) && totalTokens > 0 ? totalTokens : input + output;
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
  };
}

function enableStreamUsage(requestData) {
  if (!requestData?.stream) return;
  requestData.stream_options = {
    ...(requestData.stream_options || {}),
    include_usage: true,
  };
}

function toolCallToAssistantMessage(toolCall) {
  const message = {
    role: 'assistant',
    content: null,
    tool_calls: [{
      id: toolCall.id,
      type: 'function',
      function: {
        name: toolCall.name,
        arguments: toolCall.args || '{}',
      },
    }],
  };
  if (toolCall.reasoningContent) message.reasoning_content = toolCall.reasoningContent;
  return message;
}

function rememberToolCall(context, toolCall) {
  if (!toolCall?.id) return;
  context.pendingToolCalls?.set(toolCall.id, toolCall);
  GLOBAL_PENDING_TOOL_CALLS.set(toolCall.id, toolCall);
  if (GLOBAL_PENDING_TOOL_CALLS.size > MAX_PENDING_TOOL_CALLS) {
    const oldestKey = GLOBAL_PENDING_TOOL_CALLS.keys().next().value;
    GLOBAL_PENDING_TOOL_CALLS.delete(oldestKey);
  }
}

function getRememberedToolCall(context, callId) {
  return context.pendingToolCalls?.get(callId) || GLOBAL_PENDING_TOOL_CALLS.get(callId);
}

function toolCallReasoningFallback(message) {
  const names = (message.tool_calls || [])
    .map(tc => tc?.function?.name)
    .filter(Boolean)
    .join(', ');
  return names ? `Calling tool(s): ${names}.` : 'Calling tool to continue the task.';
}

function ensureToolCallReasoning(messages, model) {
  if (!String(model || '').startsWith('deepseek')) return messages;
  for (const message of messages || []) {
    if (message?.role !== 'assistant' || !Array.isArray(message.tool_calls) || message.tool_calls.length === 0) continue;
    if (!message.reasoning_content) message.reasoning_content = toolCallReasoningFallback(message);
  }
  return messages;
}

function hasMatchingToolCall(message, callId) {
  return message?.role === 'assistant' &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.some(tc => tc.id === callId);
}

function messageToolCallIds(message) {
  if (!Array.isArray(message?.tool_calls)) return [];
  return message.tool_calls.map(tc => tc.id).filter(Boolean);
}

function historyHasToolCall(messages, callId) {
  return messages.some(message => hasMatchingToolCall(message, callId));
}

function mergeChatMessages(history, incoming) {
  const merged = [...(history || [])];
  for (const message of incoming || []) {
    if (message.role === 'system' && merged.some(existing => existing.role === 'system' && existing.content === message.content)) {
      continue;
    }
    if (message.role === 'assistant') {
      const ids = messageToolCallIds(message);
      if (ids.length > 0 && ids.every(id => historyHasToolCall(merged, id))) continue;
    }
    if (message.role === 'tool' && merged.some(existing =>
      existing.role === 'tool' &&
      existing.tool_call_id === message.tool_call_id &&
      existing.content === message.content
    )) {
      continue;
    }
    merged.push(message);
  }
  return merged;
}

function translateRequestBody(data, context = {}) {
  if (data.model && MODEL_MAP[data.model]) {
    log(`Mapping model: ${data.model} -> ${MODEL_MAP[data.model]}`);
    data.model = MODEL_MAP[data.model];
  } else if (data.model) {
    log(`Model ${data.model} not in map, passing through`);
  }

  if (data.input !== undefined) {
    let messages = [];
    if (typeof data.input === 'string') {
      messages = [{ role: 'user', content: data.input }];
    } else if (Array.isArray(data.input)) {
      for (const item of data.input) {
        if (typeof item === 'string') {
          messages.push({ role: 'user', content: item });
          continue;
        }
        // Function call output (tool result)
        if (item.type === 'function_call_output' || item.type === 'mcp_tool_call_output') {
          const callId = item.call_id || item.id;
          const pendingToolCall = getRememberedToolCall(context, callId);
          if (pendingToolCall && !hasMatchingToolCall(messages[messages.length - 1], callId)) {
            messages.push(toolCallToAssistantMessage(pendingToolCall));
          }
          context.pendingToolCalls?.delete(callId);
          messages.push({
            role: 'tool',
            tool_call_id: callId,
            content: stringifyToolOutput(item.output),
          });
          continue;
        }
        // Function call (assistant tool call)
        if (item.type === 'function_call') {
          const callId = item.call_id || item.id;
          const pendingToolCall = getRememberedToolCall(context, callId);
          const assistantMessage = {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: callId,
              type: 'function',
              function: { name: upstreamToolNameFromCodexItem(item), arguments: item.arguments || '{}' },
            }],
          };
          const reasoningContent = item.reasoning_content || pendingToolCall?.reasoningContent;
          if (reasoningContent) assistantMessage.reasoning_content = reasoningContent;
          messages.push(assistantMessage);
          continue;
        }
        // Reasoning items - skip
        if (item.type === 'reasoning') continue;
        // Standard message
        if (item.type === 'message' || item.role) {
          messages.push({
            role: normalizeRole(item.role),
            content: flattenContent(item.content),
          });
          continue;
        }
      }
    }
    if (data.instructions) {
      messages.unshift({ role: 'system', content: data.instructions });
      delete data.instructions;
    }
    data.messages = ensureToolCallReasoning(messages, data.model);
    delete data.input;
  }

  // Translate tools from Responses to Chat Completions format
  if (data.tools) {
    data.tools = translateTools(data.tools);
  }

  // Remove Responses API-only fields
  delete data.previous_response_id;
  delete data.type;
  delete data.store;
  // Codex may return tool outputs one at a time; keep tool calls sequential for provider compatibility.
  data.parallel_tool_calls = false;
  delete data.reasoning;
  delete data.reasoning_effort;
  delete data.text;
  delete data.truncation;
  delete data.include;
  delete data.metadata;
  delete data.prompt_cache_key;
  delete data.client_metadata;
  return data;
}

function translateResponseBody(data, originalModel) {
  if (data.choices && !data.output) {
    const choice = data.choices[0] || {};
    const message = choice.message || {};
    const text = message.content || '';
    data.object = 'response';
    data.status = 'completed';
    data.output = [{
      id: 'msg_' + Date.now(),
      type: 'message',
      role: message.role || 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text: text, annotations: [] }],
    }];
    data.output_text = text;
    if (data.usage) {
      data.usage = {
        input_tokens: data.usage.prompt_tokens || 0,
        output_tokens: data.usage.completion_tokens || 0,
        total_tokens: data.usage.total_tokens || 0,
      };
    }
  }
  if (originalModel && data.model) data.model = originalModel;
  return data;
}

const OPENAI_MODELS = [
  { id: 'gpt-5.5', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'gpt-5.4', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'gpt-5', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'gpt-4o', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'gpt-4', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'gpt-4o-mini', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'o3-mini', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'o1', object: 'model', created: 1700000000, owned_by: 'openai' },
  // OpenCode Go models exposed directly
  { id: 'deepseek-v4-pro', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'deepseek-v4-flash', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'minimax-m2.7', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'minimax-m2.5', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'kimi-k2.6', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'glm-5.1', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'glm-5', object: 'model', created: 1700000000, owned_by: 'openai' },
  { id: 'qwen3.6-plus', object: 'model', created: 1700000000, owned_by: 'openai' },
];

const server = http.createServer((req, res) => {
  const url = req.url || '/';
  const pathOnly = url.split('?')[0];
  log(`${req.method} ${url} content-encoding=${req.headers['content-encoding']||'none'}`);

  // Handle GET /responses - return empty list
  if (req.method === 'GET' && (pathOnly === '/responses' || pathOnly === '/v1/responses')) {
    log('Returning empty responses list');
    const body = JSON.stringify({ object: 'list', data: [] });
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
    res.end(body);
    return;
  }

  // Handle GET /models - return OpenAI model list
  if (req.method === 'GET' && (pathOnly === '/models' || pathOnly === '/v1/models')) {
    log('Returning OpenAI model list');
    const body = JSON.stringify({ object: 'list', data: OPENAI_MODELS });
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
    res.end(body);
    return;
  }

  const chunks = [];
  req.on('data', chunk => chunks.push(chunk));

  req.on('end', () => {
    try {
      const rawBody = Buffer.concat(chunks);
      const encoding = req.headers['content-encoding'];
      const body = decompressBody(rawBody, encoding);
      const bodyStr = body.toString('utf8');

      let originalModel = null;
      let requestData = {};
      let isResponses = (pathOnly === '/responses' || pathOnly === '/v1/responses');

      if (bodyStr) {
        try {
          requestData = JSON.parse(bodyStr);
          originalModel = requestData.model;
          // Dump every request body for debugging (rotating)
          try {
            const reqDumpFile = path.join(__dirname, `req-${Date.now()}.json`);
            fs.writeFileSync(reqDumpFile, JSON.stringify(requestData, null, 2));
            fs.writeFileSync(path.join(__dirname, 'last-request.json'), JSON.stringify(requestData, null, 2));
          } catch (e) {}
        } catch (e) {
          log('JSON parse error on body:', e.message, 'first bytes:', bodyStr.substring(0, 80));
        }
      }

      let targetPath;
      let isStream = false;
      if (isResponses) {
        log('Translating Responses API -> Chat Completions');
        log('  input type:', typeof requestData.input, '| model:', requestData.model, '| stream:', requestData.stream, '| messages:', requestData.messages?.length);
        requestData = translateRequestBody(requestData);
        isStream = !!requestData.stream;
        enableStreamUsage(requestData);
        targetPath = TARGET_API_PATH + '/chat/completions';
      } else if (url.startsWith('/v1/')) {
        targetPath = TARGET_API_PATH + url.replace(/^\/v1/, '');
      } else {
        targetPath = TARGET_API_PATH + url;
      }

      log(`-> OpenCode Go ${targetPath} (model: ${requestData.model || 'unknown'}, stream: ${isStream})`);

      const proxyBody = JSON.stringify(requestData);
      try { fs.writeFileSync(path.join(__dirname, 'last-translated-request.json'), proxyBody); } catch(e) {}
      const headers = {
        'authorization': `Bearer ${TARGET_API_KEY}`,
        'accept': isStream ? 'text/event-stream' : 'application/json',
        'user-agent': 'codex-proxy',
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(proxyBody),
      };

      // For streaming Responses requests, emit response.created immediately
      // so Codex sees activity before upstream even responds.
      let earlyHeadersSent = false;
      let earlyRespId = null;
      let earlyMsgId = null;
      let heartbeatTimer = null;
      if (isStream && isResponses) {
        earlyRespId = 'resp_' + Date.now();
        earlyMsgId = 'msg_' + Date.now();
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          'connection': 'keep-alive',
          'x-accel-buffering': 'no',
        });
        res.write(`event: response.created\ndata: ${JSON.stringify({type:'response.created',response:{id:earlyRespId,object:'response',status:'in_progress',model:originalModel||'gpt-5.5',output:[]}})}\n\n`);
        earlyHeadersSent = true;
        // Heartbeat: SSE comments every 500ms keep the connection visibly alive
        // during the upstream "thinking" phase so Codex doesn't show "reconnecting".
        heartbeatTimer = setInterval(() => {
          if (!res.writableEnded) {
            try { res.write(`: keepalive ${Date.now()}\n\n`); } catch (e) {}
          }
        }, 500);
        const stopHeartbeat = () => { if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; } };
        res.on('close', stopHeartbeat);
        res.on('finish', stopHeartbeat);
      }

      const proxyReq = https.request({
        hostname: TARGET_HOST, port: 443, path: targetPath,
        method: req.method, headers, timeout: 180000,
        agent: upstreamAgent,
      }, (proxyRes) => {
        if (isStream && isResponses) {
          const respEnc = proxyRes.headers['content-encoding'];
          log(`Streaming response status=${proxyRes.statusCode} content-encoding=${respEnc || 'none'}`);
          // If non-200, capture full body to see error
          if (proxyRes.statusCode !== 200) {
            const errChunks = [];
            proxyRes.on('data', c => errChunks.push(c));
            proxyRes.on('end', () => {
              let raw = Buffer.concat(errChunks);
              raw = decompressBody(raw, respEnc);
              const errText = raw.toString('utf8');
              log('UPSTREAM ERROR BODY:', errText.slice(0, 1500));
              try { fs.writeFileSync(path.join(__dirname, 'last-error.txt'), errText); } catch(e){}
              try { fs.writeFileSync(path.join(__dirname, 'last-translated-request.json'), proxyBody); } catch(e){}
              // Headers and response.created already sent early; reuse IDs.
              const respId = earlyRespId || ('resp_' + Date.now());
              const msgId = earlyMsgId || ('msg_' + Date.now());
              if (!earlyHeadersSent) {
                res.writeHead(200, {'content-type':'text/event-stream','cache-control':'no-cache','connection':'keep-alive'});
                res.write(`event: response.created\ndata: ${JSON.stringify({type:'response.created',response:{id:respId,object:'response',status:'in_progress',model:originalModel||'gpt-5.5',output:[]}})}\n\n`);
              }
              const errMsg = `Upstream error: ${errText.slice(0,300)}`;
              res.write(`event: response.output_item.added\ndata: ${JSON.stringify({type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}})}\n\n`);
              res.write(`event: response.content_part.added\ndata: ${JSON.stringify({type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}})}\n\n`);
              res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({type:'response.output_text.delta',item_id:msgId,output_index:0,content_index:0,delta:errMsg})}\n\n`);
              res.write(`event: response.output_text.done\ndata: ${JSON.stringify({type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:errMsg})}\n\n`);
              res.write(`event: response.content_part.done\ndata: ${JSON.stringify({type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:errMsg,annotations:[]}})}\n\n`);
              res.write(`event: response.output_item.done\ndata: ${JSON.stringify({type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}})}\n\n`);
              res.write(`event: response.completed\ndata: ${JSON.stringify({type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output:[{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}],output_text:errMsg,usage:{input_tokens:0,output_tokens:0,total_tokens:0}}})}\n\n`);
              res.end();
            });
            return;
          }

          // Handle compressed streaming response using a PassThrough stream
          let upstream = proxyRes;
          if (respEnc === 'gzip' || respEnc === 'x-gzip') {
            const gunzip = zlib.createGunzip();
            proxyRes.pipe(gunzip);
            upstream = gunzip;
          } else if (respEnc === 'br') {
            const brotli = zlib.createBrotliDecompress();
            proxyRes.pipe(brotli);
            upstream = brotli;
          } else if (respEnc === 'deflate') {
            const inflate = zlib.createInflate();
            proxyRes.pipe(inflate);
            upstream = inflate;
          }

          // Headers and response.created already sent early; reuse those IDs.
          let buffer = '';
          const respId = earlyRespId || ('resp_' + Date.now());
          const msgId = earlyMsgId || ('msg_' + Date.now());
          let accumulated = '';
          let upstreamUsage = null;
          let reasoningContent = '';
          let textItemStarted = false;
          let outputIndex = 0;
          const toolCalls = {};
          const outputItems = [];

          function ensureTextItem() {
            if (textItemStarted) return;
            textItemStarted = true;
            res.write(`event: response.output_item.added\ndata: ${JSON.stringify({type:'response.output_item.added',output_index:outputIndex,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}})}\n\n`);
            res.write(`event: response.content_part.added\ndata: ${JSON.stringify({type:'response.content_part.added',item_id:msgId,output_index:outputIndex,content_index:0,part:{type:'output_text',text:'',annotations:[]}})}\n\n`);
          }

          upstream.on('data', chunk => {
            const raw = chunk.toString('utf8');
            buffer += raw;
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const payload = line.slice(5).trim();
              if (!payload || payload === '[DONE]') continue;
              try {
                const evt = JSON.parse(payload);
                if (evt.usage) upstreamUsage = evt.usage;
                if (!evt.choices || evt.choices.length === 0) continue;

                const choice = evt.choices[0];
                const delta = choice.delta;
                if (!delta) continue;
                if (delta.reasoning_content) reasoningContent += delta.reasoning_content;

                const content = delta.content;
                const tcDeltas = delta.tool_calls;

                if (content) {
                  ensureTextItem();
                  accumulated += content;
                  res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({type:'response.output_text.delta',item_id:msgId,output_index:outputIndex,content_index:0,delta:content})}\n\n`);
                }

                if (Array.isArray(tcDeltas)) {
                  for (const tc of tcDeltas) {
                    const idx = tc.index ?? 0;
                    if (!toolCalls[idx]) {
                      toolCalls[idx] = {
                        id: tc.id || ('call_' + Date.now() + '_' + idx),
                        name: '',
                        args: '',
                        itemId: 'fc_' + Date.now() + '_' + idx,
                        outputIndex: -1,
                        addedEmitted: false,
                      };
                    }
                    const t = toolCalls[idx];
                    if (tc.id) t.id = tc.id;
                    if (tc.function?.name) t.name += tc.function.name;
                    if (tc.function?.arguments) t.args += tc.function.arguments;

                    // Emit output_item.added once we have the function name
                    if (!t.addedEmitted && t.name) {
                      t.outputIndex = (textItemStarted ? 1 : 0) + Object.keys(toolCalls).filter(k => toolCalls[k].addedEmitted).length;
                      t.addedEmitted = true;
                      res.write(`event: response.output_item.added\ndata: ${JSON.stringify({type:'response.output_item.added',output_index:t.outputIndex,item:codexToolCallItem(t, '', 'in_progress')})}\n\n`);
                    }
                    if (t.addedEmitted && tc.function?.arguments) {
                      res.write(`event: response.function_call_arguments.delta\ndata: ${JSON.stringify({type:'response.function_call_arguments.delta',item_id:t.itemId,output_index:t.outputIndex,delta:tc.function.arguments})}\n\n`);
                    }
                  }
                }
              } catch (e) {
                log('Stream parse error:', e.message, 'payload:', payload.slice(0, 200));
              }
            }
          });

          upstream.on('end', () => {
            const tcCount = Object.keys(toolCalls).length;
            log(`Stream complete: ${accumulated.length} chars, ${tcCount} tool calls`);

            const finalOutputByIndex = new Map();

            if (textItemStarted) {
              res.write(`event: response.output_text.done\ndata: ${JSON.stringify({type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:accumulated})}\n\n`);
              res.write(`event: response.content_part.done\ndata: ${JSON.stringify({type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:accumulated,annotations:[]}})}\n\n`);
              res.write(`event: response.output_item.done\ndata: ${JSON.stringify({type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:accumulated,annotations:[]}]}})}\n\n`);
              finalOutputByIndex.set(0, {id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:accumulated,annotations:[]}]});
            }

            for (const idx of Object.keys(toolCalls)) {
              const t = toolCalls[idx];
              if (!t.addedEmitted) continue;
              res.write(`event: response.function_call_arguments.done\ndata: ${JSON.stringify({type:'response.function_call_arguments.done',item_id:t.itemId,output_index:t.outputIndex,arguments:t.args})}\n\n`);
              res.write(`event: response.output_item.done\ndata: ${JSON.stringify({type:'response.output_item.done',output_index:t.outputIndex,item:codexToolCallItem(t, t.args)})}\n\n`);
              finalOutputByIndex.set(t.outputIndex, codexToolCallItem(t, t.args));
              rememberToolCall({}, {id:t.id,name:t.name,args:t.args,reasoningContent});
            }

            // If nothing was emitted at all, emit an empty text item so Codex doesn't hang
            if (finalOutputByIndex.size === 0) {
              res.write(`event: response.output_item.added\ndata: ${JSON.stringify({type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}})}\n\n`);
              res.write(`event: response.content_part.added\ndata: ${JSON.stringify({type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}})}\n\n`);
              res.write(`event: response.output_text.done\ndata: ${JSON.stringify({type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:''})}\n\n`);
              res.write(`event: response.content_part.done\ndata: ${JSON.stringify({type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}})}\n\n`);
              res.write(`event: response.output_item.done\ndata: ${JSON.stringify({type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:'',annotations:[]}]}})}\n\n`);
              finalOutputByIndex.set(0, {id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:'',annotations:[]}]});
            }

            const finalOutput = [...finalOutputByIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, item]) => item);
            const usage = normalizeUsage(upstreamUsage, requestData, accumulated, toolCalls);
            res.write(`event: response.completed\ndata: ${JSON.stringify({type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output:finalOutput,output_text:accumulated,usage}})}\n\n`);
            res.end();
          });

          upstream.on('error', e => {
            log('Stream upstream error:', e.message);
            if (!res.writableEnded) {
              res.end();
            }
          });
          return;
        }

        // Non-streaming path
        const respChunks = [];
        proxyRes.on('data', c => respChunks.push(c));
        proxyRes.on('end', () => {
          let raw = Buffer.concat(respChunks);
          const respEnc = proxyRes.headers['content-encoding'];
          raw = decompressBody(raw, respEnc);
          let outBody = raw.toString('utf8');

          if (proxyRes.statusCode === 200 && outBody && isResponses) {
            try {
              const data = JSON.parse(outBody);
              outBody = JSON.stringify(translateResponseBody(data, originalModel));
            } catch (e) {
              log('Resp translate error:', e.message);
            }
          }

          const outHeaders = { ...proxyRes.headers };
          delete outHeaders['content-encoding'];
          delete outHeaders['content-length'];
          delete outHeaders['transfer-encoding'];
          outHeaders['content-type'] = 'application/json';
          outHeaders['content-length'] = Buffer.byteLength(outBody);

          log(`<- OpenCode Go ${proxyRes.statusCode} (${Buffer.byteLength(outBody)} bytes)`);
          res.writeHead(proxyRes.statusCode, outHeaders);
          res.end(outBody);
        });
      });

      proxyReq.on('error', (err) => {
        log('Proxy upstream error:', err.message);
        if (!res.headersSent) {
          res.writeHead(502, {'content-type':'application/json'});
          res.end(JSON.stringify({ error: { message: err.message } }));
        } else if (!res.writableEnded) {
          // Early headers already sent for streaming; emit a complete text item so Codex can close cleanly.
          const respId = earlyRespId || ('resp_' + Date.now());
          const msgId = earlyMsgId || ('msg_' + Date.now());
          const errMsg = `Upstream connection error: ${err.message}`;
          res.write(`event: response.output_item.added\ndata: ${JSON.stringify({type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}})}\n\n`);
          res.write(`event: response.content_part.added\ndata: ${JSON.stringify({type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}})}\n\n`);
          res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({type:'response.output_text.delta',item_id:msgId,output_index:0,content_index:0,delta:errMsg})}\n\n`);
          res.write(`event: response.output_text.done\ndata: ${JSON.stringify({type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:errMsg})}\n\n`);
          res.write(`event: response.content_part.done\ndata: ${JSON.stringify({type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:errMsg,annotations:[]}})}\n\n`);
          res.write(`event: response.output_item.done\ndata: ${JSON.stringify({type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}})}\n\n`);
          res.write(`event: response.completed\ndata: ${JSON.stringify({type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output:[{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}],output_text:errMsg,usage:{input_tokens:0,output_tokens:0,total_tokens:0}}})}\n\n`);
          res.end();
        }
      });

      proxyReq.write(proxyBody);
      proxyReq.end();

    } catch (err) {
      log('Handler error:', err.message);
      if (!res.headersSent) {
        res.writeHead(500, {'content-type':'application/json'});
        res.end(JSON.stringify({ error: { message: err.message } }));
      }
    }
  });
});

function encodeWsFrame(text) {
  const payload = Buffer.from(text, 'utf8');
  let header;
  if (payload.length < 126) {
    header = Buffer.from([0x81, payload.length]);
  } else if (payload.length < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  return Buffer.concat([header, payload]);
}

function sendWsJson(socket, event) {
  if (!socket.destroyed) socket.write(encodeWsFrame(JSON.stringify(event)));
}

function decodeWsFrames(state, chunk) {
  state.buffer = Buffer.concat([state.buffer, chunk]);
  const frames = [];
  let offset = 0;

  while (state.buffer.length - offset >= 2) {
    const first = state.buffer[offset];
    const second = state.buffer[offset + 1];
    const opcode = first & 0x0f;
    const masked = (second & 0x80) !== 0;
    let length = second & 0x7f;
    let headerLen = 2;

    if (length === 126) {
      if (state.buffer.length - offset < 4) break;
      length = state.buffer.readUInt16BE(offset + 2);
      headerLen = 4;
    } else if (length === 127) {
      if (state.buffer.length - offset < 10) break;
      const bigLength = state.buffer.readBigUInt64BE(offset + 2);
      if (bigLength > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('WebSocket frame too large');
      length = Number(bigLength);
      headerLen = 10;
    }

    const maskLen = masked ? 4 : 0;
    const frameLen = headerLen + maskLen + length;
    if (state.buffer.length - offset < frameLen) break;

    let payload = state.buffer.subarray(offset + headerLen + maskLen, offset + frameLen);
    if (masked) {
      const mask = state.buffer.subarray(offset + headerLen, offset + headerLen + 4);
      payload = Buffer.from(payload);
      for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i % 4];
    }

    frames.push({ opcode, payload });
    offset += frameLen;
  }

  state.buffer = state.buffer.subarray(offset);
  return frames;
}

function extractWsRequest(parsed) {
  if (parsed && parsed.model && (parsed.input !== undefined || parsed.messages)) return parsed;
  if (parsed && parsed.body) return typeof parsed.body === 'string' ? JSON.parse(parsed.body) : parsed.body;
  if (parsed && parsed.request) return extractWsRequest(parsed.request);
  if (parsed && parsed.params) return extractWsRequest(parsed.params);
  return null;
}

function handleResponsesWsRequest(socket, incoming, context) {
  let requestData = extractWsRequest(incoming);
  if (!requestData) {
    log('WS request shape not recognized:', JSON.stringify(incoming).slice(0, 500));
    return;
  }

  const originalModel = requestData.model;
  log('WS /responses request model:', originalModel, '| stream:', requestData.stream, '| input:', typeof requestData.input);
  requestData = translateRequestBody(requestData, context);
  const currentMessages = requestData.messages || [];
  requestData.messages = ensureToolCallReasoning(mergeChatMessages(context.chatHistory, currentMessages), requestData.model);
  requestData.stream = true;
  enableStreamUsage(requestData);
  const proxyBody = JSON.stringify(requestData);
  try { fs.writeFileSync(path.join(__dirname, 'last-translated-request.json'), proxyBody); } catch(e) {}

  const respId = 'resp_' + Date.now();
  const msgId = 'msg_' + Date.now();
  sendWsJson(socket, {type:'response.created',response:{id:respId,object:'response',status:'in_progress',model:originalModel||'gpt-5.5',output:[]}});

  const proxyReq = https.request({
    hostname: TARGET_HOST,
    port: 443,
    path: TARGET_API_PATH + '/chat/completions',
    method: 'POST',
    headers: {
      'authorization': `Bearer ${TARGET_API_KEY}`,
      'accept': 'text/event-stream',
      'user-agent': 'codex-proxy',
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(proxyBody),
    },
    timeout: 180000,
    agent: upstreamAgent,
  }, (proxyRes) => {
    const respEnc = proxyRes.headers['content-encoding'];
    log(`WS streaming response status=${proxyRes.statusCode} content-encoding=${respEnc || 'none'}`);

    if (proxyRes.statusCode !== 200) {
      const errChunks = [];
      proxyRes.on('data', c => errChunks.push(c));
      proxyRes.on('end', () => {
        let raw = decompressBody(Buffer.concat(errChunks), respEnc);
        const errMsg = `Upstream error: ${raw.toString('utf8').slice(0, 300)}`;
        sendWsJson(socket, {type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}});
        sendWsJson(socket, {type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}});
        sendWsJson(socket, {type:'response.output_text.delta',item_id:msgId,output_index:0,content_index:0,delta:errMsg});
        sendWsJson(socket, {type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:errMsg});
        sendWsJson(socket, {type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:errMsg,annotations:[]}});
        sendWsJson(socket, {type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}});
        sendWsJson(socket, {type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output:[{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:errMsg,annotations:[]}]}],output_text:errMsg,usage:{input_tokens:0,output_tokens:0,total_tokens:0}}});
      });
      return;
    }

    let upstream = proxyRes;
    if (respEnc === 'gzip' || respEnc === 'x-gzip') upstream = proxyRes.pipe(zlib.createGunzip());
    else if (respEnc === 'br') upstream = proxyRes.pipe(zlib.createBrotliDecompress());
    else if (respEnc === 'deflate') upstream = proxyRes.pipe(zlib.createInflate());

    let buffer = '';
    let accumulated = '';
    let upstreamUsage = null;
    let reasoningContent = '';
    let textItemStarted = false;
    const toolCalls = {};
    const outputItems = new Map();

    function ensureTextItem() {
      if (textItemStarted) return;
      textItemStarted = true;
      sendWsJson(socket, {type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}});
      sendWsJson(socket, {type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}});
    }

    upstream.on('data', chunk => {
      buffer += chunk.toString('utf8');
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const evt = JSON.parse(payload);
          if (evt.usage) upstreamUsage = evt.usage;
          const delta = evt.choices?.[0]?.delta;
          if (!delta) continue;
          if (delta.reasoning_content) reasoningContent += delta.reasoning_content;
          if (delta.content) {
            ensureTextItem();
            accumulated += delta.content;
            sendWsJson(socket, {type:'response.output_text.delta',item_id:msgId,output_index:0,content_index:0,delta:delta.content});
          }
          if (Array.isArray(delta.tool_calls)) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0;
              if (!toolCalls[idx]) {
                toolCalls[idx] = {id:tc.id || ('call_' + Date.now() + '_' + idx), name:'', args:'', itemId:'fc_' + Date.now() + '_' + idx, outputIndex:-1, added:false};
              }
              const t = toolCalls[idx];
              if (tc.id) t.id = tc.id;
              if (tc.function?.name) t.name += tc.function.name;
              if (tc.function?.arguments) t.args += tc.function.arguments;
              if (!t.added && t.name) {
                t.outputIndex = (textItemStarted ? 1 : 0) + Object.values(toolCalls).filter(v => v.added).length;
                t.added = true;
                sendWsJson(socket, {type:'response.output_item.added',output_index:t.outputIndex,item:codexToolCallItem(t, '', 'in_progress')});
              }
              if (t.added && tc.function?.arguments) {
                sendWsJson(socket, {type:'response.function_call_arguments.delta',item_id:t.itemId,output_index:t.outputIndex,delta:tc.function.arguments});
              }
            }
          }
        } catch (e) {
          log('WS stream parse error:', e.message, 'payload:', payload.slice(0, 200));
        }
      }
    });

    upstream.on('end', () => {
      if (textItemStarted) {
        sendWsJson(socket, {type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:accumulated});
        sendWsJson(socket, {type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:accumulated,annotations:[]}});
        sendWsJson(socket, {type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:accumulated,annotations:[]}]}});
        outputItems.set(0, {id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:accumulated,annotations:[]}]});
      }
      for (const t of Object.values(toolCalls)) {
        if (!t.added) continue;
        sendWsJson(socket, {type:'response.function_call_arguments.done',item_id:t.itemId,output_index:t.outputIndex,arguments:t.args});
        sendWsJson(socket, {type:'response.output_item.done',output_index:t.outputIndex,item:codexToolCallItem(t, t.args)});
        outputItems.set(t.outputIndex, codexToolCallItem(t, t.args));
        rememberToolCall(context, {id:t.id,name:t.name,args:t.args,reasoningContent});
      }
      if (Object.values(toolCalls).some(t => t.added)) {
        log('WS tool calls:', Object.values(toolCalls).filter(t => t.added).map(t => `${t.name}->${codexToolDisplayName(t.name)}`).join(', '));
      }
      context.chatHistory = mergeChatMessages(context.chatHistory, currentMessages);
      if (textItemStarted && !Object.values(toolCalls).some(t => t.added)) {
        context.chatHistory.push({ role: 'assistant', content: accumulated });
      }
      if (outputItems.size === 0) {
        sendWsJson(socket, {type:'response.output_item.added',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'in_progress',content:[]}});
        sendWsJson(socket, {type:'response.content_part.added',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}});
        sendWsJson(socket, {type:'response.output_text.done',item_id:msgId,output_index:0,content_index:0,text:''});
        sendWsJson(socket, {type:'response.content_part.done',item_id:msgId,output_index:0,content_index:0,part:{type:'output_text',text:'',annotations:[]}});
        sendWsJson(socket, {type:'response.output_item.done',output_index:0,item:{id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:'',annotations:[]}]}});
        outputItems.set(0, {id:msgId,type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:'',annotations:[]}]});
      }
      const output = [...outputItems.entries()].sort((a, b) => a[0] - b[0]).map(([, item]) => item);
      const usage = normalizeUsage(upstreamUsage, requestData, accumulated, toolCalls);
      sendWsJson(socket, {type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output,output_text:accumulated,usage}});
      log(`WS stream complete: ${accumulated.length} chars, ${Object.keys(toolCalls).length} tool calls`);
    });
  });

  proxyReq.on('error', err => {
    const errMsg = `Upstream connection error: ${err.message}`;
    log('WS proxy upstream error:', err.message);
    sendWsJson(socket, {type:'response.completed',response:{id:respId,object:'response',status:'completed',model:originalModel||'gpt-5.5',output:[],output_text:errMsg,usage:{input_tokens:0,output_tokens:0,total_tokens:0}}});
  });

  proxyReq.write(proxyBody);
  proxyReq.end();
}

server.on('upgrade', (req, socket, head) => {
  const pathOnly = (req.url || '/').split('?')[0];
  if (pathOnly !== '/responses' && pathOnly !== '/v1/responses') {
    socket.write('HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
    socket.destroy();
    return;
  }

  log(`WS UPGRADE: ${req.method} ${req.url} - accepting Responses websocket`);
  const crypto = require('crypto');
  const wsKey = req.headers['sec-websocket-key'];
  if (!wsKey) {
    socket.write('HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
    socket.destroy();
    return;
  }

  const accept = crypto.createHash('sha1')
    .update(wsKey + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
    .digest('base64');
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n' +
    'Upgrade: websocket\r\n' +
    'Connection: Upgrade\r\n' +
    `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
  );

  const state = { buffer: Buffer.alloc(0) };
  const context = { pendingToolCalls: new Map() };
  const keepalive = setInterval(() => {
    if (!socket.destroyed) socket.write(Buffer.from([0x89, 0x00]));
  }, 25000);
  const cleanup = () => clearInterval(keepalive);
  socket.on('close', cleanup);
  socket.on('error', cleanup);
  socket.on('data', chunk => {
    try {
      for (const frame of decodeWsFrames(state, chunk)) {
        if (frame.opcode === 0x8) {
          socket.write(Buffer.from([0x88, 0x00]));
          socket.end();
        } else if (frame.opcode === 0x9) {
          socket.write(Buffer.from([0x8a, 0x00]));
        } else if (frame.opcode === 0x1 || frame.opcode === 0x2) {
          const text = frame.payload.toString('utf8');
          log('WS request frame:', text.slice(0, 500));
          const parsed = JSON.parse(text);
          if (!parsed.type || parsed.type === 'response.create' || parsed.model || parsed.body || parsed.request || parsed.params) {
            handleResponsesWsRequest(socket, parsed, context);
          }
        }
      }
    } catch (e) {
      log('WS frame handling error:', e.message);
    }
  });
});

server.listen(8080, '127.0.0.1', () => {
  log('HTTP proxy running on http://127.0.0.1:8080 (with gzip + streaming support)');
});
