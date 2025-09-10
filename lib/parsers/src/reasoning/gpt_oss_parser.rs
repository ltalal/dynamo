// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::ParserResult;
use crate::ReasoningParser;

use openai_harmony::StreamableParser;
use openai_harmony::chat::TextContent;
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, chat::Role, load_harmony_encoding};

///// Static initialization of harmony encoder to not affect performance every time a parser is created
/// This is because load_harmony_encoding downloads some tiktoken files into a directory and we don't want to do this every time we create a parser.
use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();

fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
}

pub struct GptOssReasoningParser {
    parser: StreamableParser,
    generated_text_processed: bool,
}

/// Implement Debug for GptOssReasoningParser separately because StreamableParser does not implement Debug
impl Debug for GptOssReasoningParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptOssReasoningParser")
            .field("parser", &self.parser.state_json())
            .finish()
    }
}

impl GptOssReasoningParser {
    pub fn new() -> anyhow::Result<Self> {
        let parser = match get_harmony_encoding().as_ref() {
            Ok(enc) => match StreamableParser::new(enc.clone(), Some(Role::Assistant)) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Harmony StreamableParser init failed for GPT OSS: {e}");
                    return Err(anyhow::anyhow!(
                        "Failed to load Harmony StreamableParser: {e}"
                    ));
                }
            },
            Err(e) => {
                tracing::warn!("Failed to load Harmony encoding for GPT OSS: {e}");
                return Err(anyhow::anyhow!("Failed to load Harmony encoding: {e}"));
            }
        };
        Ok(Self {
            parser,
            generated_text_processed: false,
        })
    }
}

impl ReasoningParser for GptOssReasoningParser {
    fn detect_and_parse_reasoning(&mut self, _text: &str, token_ids: &[u32]) -> ParserResult {
        tracing::debug!(
            "detect_and_parse_reasoning called with {} token_ids",
            token_ids.len()
        );

        let parser = &mut self.parser;

        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
            }
        }

        let output_msgs = parser.messages();
        tracing::debug!("Parser has {} output messages", output_msgs.len());

        match output_msgs.len() {
            0 => {
                tracing::debug!("No output messages, using current content");
                let current = parser.current_content().unwrap_or_default();
                tracing::debug!("Current content length: {}", current.len());
                ParserResult {
                    normal_text: String::new(),
                    reasoning_text: current,
                }
            }
            1 => {
                tracing::debug!("Single output message detected");
                let mut reasoning_text = String::new();
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    output_msgs[0].content.first()
                {
                    reasoning_text.push_str(text);
                    tracing::debug!("Extracted reasoning text length: {}", reasoning_text.len());
                }
                let current = parser.current_content().unwrap_or_default();
                tracing::debug!("Current content length: {}", current.len());
                ParserResult {
                    normal_text: current,
                    reasoning_text,
                }
            }
            _ => {
                tracing::debug!("Multiple output messages detected: {}", output_msgs.len());
                let mut reasoning_text = String::new();
                let mut normal_text = String::new();

                // Loop until second last message
                for (i, parse_msg) in output_msgs.iter().take(output_msgs.len() - 1).enumerate() {
                    tracing::debug!("Processing reasoning message {}", i + 1);
                    if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                        parse_msg.content.first()
                    {
                        reasoning_text.push_str(text);
                        tracing::debug!("Added {} chars to reasoning text", text.len());
                    }
                }

                let last_msg = &output_msgs[output_msgs.len() - 1];
                tracing::debug!("Processing final message");

                // Handle the last message
                if let Some(openai_harmony::chat::Content::Text(TextContent { text })) =
                    last_msg.content.first()
                {
                    normal_text.push_str(text);
                    tracing::debug!("Added {} chars to normal text", text.len());
                }

                tracing::debug!(
                    "Final result - normal_text: {} chars, reasoning_text: {} chars",
                    normal_text.len(),
                    reasoning_text.len()
                );

                ParserResult {
                    normal_text,
                    reasoning_text,
                }
            }
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        _text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        tracing::debug!(
            "parse_reasoning_streaming_incremental called with {} token_ids",
            token_ids.len()
        );

        let parser: &mut StreamableParser = &mut self.parser;
        for (i, token_id) in token_ids.iter().enumerate() {
            tracing::debug!(
                "Processing streaming token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            if let Err(e) = parser.process(*token_id) {
                tracing::warn!("Harmony parse error for token_id {token_id}: {e}");
                return ParserResult::default();
            }
        }

        // Debug: append tokens and text to two files
        // Write tokens to "tokens.txt" and text to "text.txt"
        use std::io::Write;

        // Convert token_ids to string for debugging
        let tokens_str = token_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Write tokens to file
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("tokens.txt")
            .and_then(|mut file| writeln!(file, "{}", tokens_str))
        {
            tracing::warn!("Failed to write tokens to file: {}", e);
        }

        // Write text to file
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("text.txt")
            .and_then(|mut file| writeln!(file, "{}", _text))
        {
            tracing::warn!("Failed to write text to file: {}", e);
        }
        // write the state json to file
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("state.json")
            .and_then(|mut file| match self.parser.state_json() {
                Ok(state_json) => writeln!(file, "{}", state_json),
                Err(json_err) => {
                    tracing::warn!("Failed to serialize parser state: {}", json_err);
                    writeln!(
                        file,
                        "{{\"error\": \"Failed to serialize state: {}\"}}",
                        json_err
                    )
                }
            })
        {
            tracing::warn!("Failed to write state json to file: {}", e);
        }

        if let Some(channel) = self.parser.current_channel() {
            tracing::debug!("Current channel {}", channel);
            if channel == "final" {
                tracing::debug!("In final channel, processing normal text");
                // If we're in the final channel, we should not parse reasoning
                if let Some(current) = self.parser.last_content_delta().unwrap_or_default() {
                    tracing::debug!("Got normal text delta of {} chars", current.len());
                    return ParserResult {
                        normal_text: current,
                        reasoning_text: String::new(),
                    };
                }
                tracing::debug!("No content delta in final channel");
                ParserResult::default()
            } else if channel == "commentary" {
                tracing::debug!(
                    "In final/commentary channel, returning raw token content for tool parser"
                );
                // If we're in the final or commentary channel, we should return raw token content
                // so that the tool parser can process it properly
                if let Ok(enc) = get_harmony_encoding() {
                    // Use the tokenizer to decode token IDs back to text
                    // Since we're in final/commentary channel, we want the raw content for tool parser
                    // let raw_content = String::from_utf8_lossy(&raw_bytes).to_string();
                    let raw_content = self.parser.current_content().unwrap_or_default();
                    let mut final_text = _text.to_string();

                    // Only process generated text once, the first time we encounter this condition
                    if !self.generated_text_processed {
                        eprintln!("tokens {:?}", self.parser.tokens());
                        eprintln!("parser.messages.len() {:?}", self.parser.messages().len());

                        let last_msg = self.parser.messages().last().unwrap();
                        eprintln!("last_msg {:?}", last_msg);

                        // iterate over the tokens and find the last <|start|> 200005 from tail to head
                        let tokens = self.parser.tokens();
                        let mut last_start_token = None;

                        for (i, token) in tokens.iter().enumerate().rev() {
                            if token == &200005 {
                                last_start_token = Some(i);
                                break;
                            }
                        }

                        let last_start_token = last_start_token.unwrap_or(0);
                        // then get the generate text between the last <|start|> to the end of self.parser.tokens()
                        let end_token = self.parser.tokens().len();
                        // using the harmony decode_utf8 to translate the tokens to text
                        let generated_text = enc
                            .tokenizer()
                            .decode_utf8(&self.parser.tokens()[last_start_token..end_token])
                            .unwrap();

                        // let generated_text = format!("<|start|>assistant{}", generated_text);
                        eprintln!("generated_text {:?}", generated_text);
                        final_text = generated_text;

                        // Mark as processed to prevent running this again
                        self.generated_text_processed = true;
                    }

                    tracing::debug!("Got raw token content of {} chars", raw_content.len());
                    tracing::debug!("Raw content[!!!!!!]:: {}", raw_content);
                    tracing::debug!("_text[!!!!!!]:: {}", _text);
                    ParserResult {
                        normal_text: final_text,
                        reasoning_text: String::new(),
                    }
                } else {
                    tracing::warn!("Failed to get harmony encoding for raw token decoding");
                    ParserResult::default()
                }
            } else {
                tracing::debug!("In reasoning channel: {}", channel);
                if let Some(current) = self.parser.last_content_delta().unwrap_or_default() {
                    tracing::debug!("Got reasoning text delta of {} chars", current.len());
                    return ParserResult {
                        normal_text: String::new(),
                        reasoning_text: current,
                    };
                }
                tracing::debug!("No content delta in reasoning channel");
                ParserResult::default()
            }
        } else {
            tracing::debug!("No current channel detected");
            ParserResult::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_oss_reasoning_parser() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text); // Example token IDs
        let result = parser.detect_and_parse_reasoning("Test text", &token_ids);
        assert!(result.normal_text == "The capital of Brazil is Brasília.");
        assert!(
            result.reasoning_text
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text); // Example token IDs
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for token in token_ids.iter() {
            let result = parser.parse_reasoning_streaming_incremental("Test text", &[(*token)]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert!(normal_text_incr == "The capital of Brazil is Brasília.");
        assert!(
            reasoning_text_incr
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }
}
