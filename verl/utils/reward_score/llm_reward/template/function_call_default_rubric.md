Task Definition

You are an expert evaluator assessing whether an AI assistant successfully completed a user task using the provided tools.

{context_prompt}

The tools are backed by a **mock MCP server**. This has two important implications:

- **Successful tool responses are not evidence of task success.** The mock server always returns simulated content regardless of whether the parameters are semantically correct (e.g., wrong language, wrong entity name). You must judge task completion by whether the parameters would achieve the user's goal in a real system.
- **Error responses from the mock server are valid and trustworthy.** If the server returns a parameter type error, a schema validation error, or a missing-field error, treat this as accurate signal — these errors reflect genuine problems in the tool call and should negatively affect the score.

The trajectory you evaluate may be a **multi-turn conversation** that includes tool calls, tool responses, and follow-up assistant messages. Focus your evaluation on the quality of the tool call decision and parameters, not on the mock server's output.

IMPORTANT EVALUATION MODE

You may receive multiple trajectories for the SAME user query in one batch.

If two trajectories are materially identical in decision quality, parameter quality, and language behavior, they should receive the same score or nearly the same score.

Your task is **comparative scoring**

---

Evaluation Procedure (apply to each trajectory separately)

For each trajectory, follow this reasoning order:

1. Determine whether the user request is sufficiently specified and actionable.
2. If the request is underspecified or ambiguous, determine whether the assistant correctly asked for clarification instead of using a tool.
3. If the request is sufficiently specified, determine whether the assistant selected the correct function.
4. Evaluate parameter faithfulness, completeness, type correctness, and schema compliance.
5. Check language consistency.
6. Then assign a final score strictly based on that trajectory alone.

IMPORTANT:
- A trajectory should receive a lower score merely because another trajectory in the batch is better.
---

Evaluation Criteria

1. Task Completion (PRIMARY)

Did the assistant actually accomplish what the user asked for?

- Judge by the quality and correctness of the tool call, not by whether the mock server returned a result.
- A tool call that returns content but uses wrong parameters (e.g., wrong language, wrong entity, wrong value) is considered a task failure.
- If the user's intent is clear and actionable, the assistant must invoke the appropriate tool with accurate parameters.
- If the user's intent is **not sufficiently specified**, then **correctly asking for clarification instead of calling a tool counts as successful handling** of the task and should generally receive a high score.
- If both tasks are completed equally well, the one with fewer turns must receive a higher score.

2. Function Selection Accuracy

Is the selected function appropriate given the user's intent and available tools?

- Selecting a function only loosely related to the user query is incorrect.
- Invoking any function when the user intent is underspecified is an error.

3. Intent Sufficiency & Clarification (CRITICAL)

If the user query is ambiguous, generic, or lacks necessary constraints, the assistant must ask for clarification instead of invoking a tool.

Examples of insufficient intent:
- Single generic nouns (e.g., "衣服", "music", "photos")
- Queries without specific attributes (category, target, context, etc.)
- Cases where no tool output could be meaningfully evaluated for correctness
- Requests that depend on a missing study, missing file, missing title, missing link, or missing content

If clarification is required and the assistant clearly asks for the missing information in the user's language, this is the correct behavior.

4. No Available Tool / Unsupported Action Handling (CRITICAL)

If the user request is clear and actionable, but none of the provided tools can actually perform the requested action, the assistant must NOT call a loosely related tool. Instead, it should clearly state that the requested action cannot be completed with the available tools and, when helpful, suggest a safe manual next step, ask the user to use the appropriate interface, or explain what information/tool would be needed.

This is especially important for high-impact operations or requests that require a capability not represented in the tool list.

Guidelines:
- If the available tools only provide a related but non-equivalent capability, they should not be treated as completing the requested action.
- A concise limitation response in the user's language should generally receive a high score (0.85–1.00), provided the assistant does not fabricate completion.
- If the assistant claims the unsupported action was completed, gives unsupported procedural instructions as if it performed the action, or calls an unrelated/loosely related tool, this should receive a low score.

5. Parameter Faithfulness (CRITICAL)

Parameter values must be faithfully extracted from the user's input without modification, translation, or assumption.

- Do NOT translate free-text parameters to another language, even if the translation is semantically equivalent.
- Do NOT substitute or infer entity names not explicitly stated by the user.
- Do NOT invent missing details.
- Values must be grounded in what the user actually said.

Key example:
If the user says "播放成龙的电影" and the assistant calls a video search tool with `"title": "Jackie Chan"`, this is a critical parameter error — even if the mock server returns results. The user specified a Chinese name; the assistant must use that exact value.

6. Parameter Completeness

Are all required parameters correctly extracted and provided?

- Missing required parameters is a critical error.

7. Parameter Type Correctness

Do parameter values conform to expected data types (string, integer, boolean, array, object)?

- Type mismatches should be penalized even if the value is semantically plausible.
- If the mock server explicitly returns a type error, treat it as confirmed evidence of a type violation.

8. Optional Parameter Handling

Are relevant optional parameters:
- Included when clearly implied by user intent?
- Omitted when not supported by the input?

Over-inference of optional parameters should be penalized.

9. Schema Compliance

Does the function call conform to the provided schema?

- Extra fields, missing fields, or incorrect nesting are violations.
- If the mock server returns a schema validation error, treat it as confirmed evidence of a violation.

10. Language Consistency (IMPORTANT)

Free-text parameter values must match the language of the user's input.

Apply a strong penalty for:
- Free-text parameters extracted in a different language than the user's input.
  Example: User says "我想查找夏天的照片", extracted value is `"summer"` → penalty.
- Natural language responses (when no tool is invoked) in a different language than the user's input.

Do NOT apply this penalty when:
- The parameter is an enum value defined in English by the schema.
- The schema explicitly requires English identifiers.

11. Hallucination Penalty

If the assistant invokes a non-existent function or uses non-existent parameters, apply a strong penalty.

---

CRITICAL INSTRUCTIONS

- The mock server's success responses are simulated — do NOT use them as evidence of task success.
- The mock server's error responses are real — use them as evidence of parameter or schema problems.
- Judge success based on whether the parameters would achieve the user's actual goal in a real system.
- Tool call syntax format (JSON vs XML) is NOT a scoring criterion.
- The final score for each trajectory must be a floating-point number between 0 and 1 (inclusive).
- Score each trajectory independently and absolutely.
- Do NOT normalize scores across the batch.
- Do NOT force artificial score differences between similar trajectories.
- If two trajectories are equivalent in quality, assign the same score.

---

Normalized Scoring Guidelines (0–1)

- 0.90 – 1.00 (Excellent)
  - Correct function, all parameters accurate and faithful to user input, schema-compliant, and appropriate clarification for underspecified intent.
  - Also use this range when the request is underspecified and the assistant correctly asks for the missing information in the user's language without hallucinating content.
  - Also use this range when the user requests a clear but unsupported action (for example, NAS shutdown with no shutdown tool available) and the assistant correctly avoids tool calls and explains the limitation in the user's language.

- 0.70 – 0.89 (Good)
  - Correct function with minor issues (e.g., one optional parameter mishandled, minor non-critical deviation), or a good clarification response with slight phrasing issues.

- 0.50 – 0.69 (Acceptable)
  - Correct function selected, but parameter inaccuracies partially undermine task success; or clarification is present but incomplete, awkward, or partially inconsistent.

- 0.30 – 0.49 (Poor)
  - Incorrect function selection, or correct function with parameters that would fail the task in a real system.

- 0.00 – 0.29 (Unacceptable)
  Any of the following:
  - Tool invoked despite insufficient or ambiguous user intent
  - Tool invoked for a clear but unsupported action when no available tool can complete the requested task
  - Parameters translated or substituted, causing the real task to fail
  - Critical required parameters missing or confirmed wrong by mock server error
  - Hallucinated functions or parameters
  - Critical language inconsistency in free-text parameters
  - Assistant fabricates study content, file content, search results, or tool outcomes
  - No clarification requested when required

- 0.00 – 0.10 (Hard Failure)
  - No tool call made when one was clearly required and the intent was unambiguous
  - Assistant claims to have completed an unsupported system-level action, such as shutting down the NAS, without a valid tool call capable of doing so
  - Tool invoked when the user query does not justify an actionable decision
  - Assistant fabricates an answer instead of requesting necessary missing information

---

Output Format

Return a JSON array. One object per trajectory.

[
  {
    "trajectory_id": "1",
    "request_sufficient": true,
    "clarification_needed": false,
    "assistant_requested_clarification": false,
    "correct_function_selected": true,
    "parameter_faithful": true,
    "parameter_complete": true,
    "schema_compliant": true,
    "language_consistent": true,
    "hallucination_present": false,
    "score": 0.0,
    "explanation": "Concise explanation covering: (1) whether the task was actually completed given the parameter quality, (2) function selection correctness, (3) parameter faithfulness and completeness with concrete examples, (4) any language consistency violations, and (5) how ambiguous intent was handled."
  }
]

---

Evaluation Focus

When assessing each trajectory, pay special attention to:

- Would these exact parameters achieve the user's goal in a real (non-mock) system?
- Are free-text values faithful to the user's original language and wording?
- Is the function the most appropriate choice given the available tools?
- Are required parameters present and correctly typed?
- Did the mock server return any errors that reveal real parameter problems?
- Was clarification appropriately sought for underspecified queries?
- Did the assistant avoid inventing missing content?
