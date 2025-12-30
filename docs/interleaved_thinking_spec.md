# Specification: Interleaved Thinking API

## 1. Overview

The Interleaved Thinking feature allows a client to receive a structured stream of events that distinguishes between a large language model's (LLM) internal "thought process" and its final answer. This is achieved by having the server parse a single, continuous stream of text from the LLM for special XML-like tags (`<thinking>` and `</thinking>`).

This feature is only available for streaming requests (`stream: true`).

## 2. Model Requirements

To use this feature, the LLM must be prompted or fine-tuned to follow a specific output format:
1.  It may optionally output some introductory text.
2.  It must enclose its reasoning, planning, or thought process within `<thinking>` and `</thinking>` tags.
3.  After the closing `</thinking>` tag, it should output the final, user-facing answer.

**Example of expected model output (as a single stream):**

```
I need to answer the user's question about the first three letters of the alphabet. <thinking>Step 1: Identify the user's core question. The user wants the first 3 letters of the English alphabet. Step 2: Recall the sequence of the alphabet. It starts with A, B, C. Step 3: Formulate the final answer.</thinking>The first three letters of the alphabet are A, B, and C.
```

## 3. Server-Side Parsing Logic

The API server maintains a state machine that parses the text stream from the model in real-time.

-   **States**: `text`, `thinking`
-   **Buffer**: Accumulates incoming text chunks to correctly identify complete tags.
-   **Content Block Index**: An integer (`index`) that increments for each new content block.

The server transitions between states based on the detection of the `<thinking>` and `</thinking>` tags, generating a sequence of Server-Sent Events (SSE) that are compliant with the Anthropic Messages API specification.

## 4. SSE Event Sequence Example

This section details the exact sequence of SSE events the client will receive when the model produces the example output from Section 2.

### Step 1: Message Start

The server always begins by sending a `message_start` event. This signals the beginning of the entire response.

```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_...", "type": "message", "role": "assistant", ...}}
```

### Step 2: First Text Block (Pre-Thinking)

The server immediately opens the first content block (`index: 0`) for the initial part of the answer.

```
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
```

It then sends the text preceding the `<thinking>` tag as one or more `content_block_delta` events.

```
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I need to answer the user's question about the first three letters of the alphabet. "}}
```

### Step 3: Transition to Thinking Block

Upon detecting the `<thinking>` tag, the server closes the first block and opens a new one for the thought process.

1.  **Stop the first text block:**
    ```
    event: content_block_stop
    data: {"type": "content_block_stop", "index": 0}
    ```
2.  **Start the thinking block (`index: 1`):**
    ```
    event: content_block_start
    data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}
    ```

### Step 4: Streaming the Thought Process

The text between the `<thinking>` and `</thinking>` tags is streamed as `content_block_delta` events within the new block (`index: 1`).

```
event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Step 1: Identify the user's core question. The user wants the first 3 letters of the English alphabet. "}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Step 2: Recall the sequence of the alphabet. It starts with A, B, C. "}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Step 3: Formulate the final answer."}}
```

### Step 5: Transition to Final Answer Block

Upon detecting the `</thinking>` tag, the server closes the thinking block and opens a final block for the answer.

1.  **Stop the thinking block:**
    ```
    event: content_block_stop
    data: {"type": "content_block_stop", "index": 1}
    ```
2.  **Start the final answer block (`index: 2`):**
    ```
    event: content_block_start
    data: {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}
    ```

### Step 6: Streaming the Final Answer

The text following the `</thinking>` tag is streamed as `content_block_delta` events in the final block (`index: 2`).

```
event: content_block_delta
data: {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "The first three letters of the alphabet are A, B, and C."}}
```

### Step 7: Message Stop

Once the model finishes generating text, the server closes the final block and then signals the end of the entire message.

1.  **Stop the final block:**
    ```
    event: content_block_stop
    data: {"type": "content_block_stop", "index": 2}
    ```
2.  **End the message:**
    ```
    event: message_stop
    data: {"type": "message_stop", ...}
    ```
