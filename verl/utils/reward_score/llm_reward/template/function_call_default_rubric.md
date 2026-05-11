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
- A trajectory should NOT receive a lower score merely because another trajectory in the batch is better.
---

Evaluation Criteria

Pre-Scoring Validity Gates (apply before assigning the final score)

Use these gates to avoid giving high scores to incoherent rollouts that happen to contain a plausible fragment.

- If the assistant output is mostly gibberish, unrelated multilingual fragments, random code-like tokens, or otherwise not a coherent response to the user, the score must be 0.00–0.20 even if a few words look relevant.
- If the assistant output contains a malformed, unparseable, or unfinished tool call, the score must be 0.00–0.29 unless there is also a separate valid tool call that fully completes the task.
- If the assistant invokes a non-existent function, a non-existent parameter, or a related but non-equivalent tool for the user's request, the score must be 0.00–0.29.
- If a clear and actionable request requires a tool call but no valid tool call is made, the score must be 0.00–0.10.
- If the tool call is correct but the assistant later contradicts the action, fabricates additional results, or adds substantially incoherent follow-up text, cap the score at 0.70. If the follow-up text is only a minor awkward confirmation and does not change the completed action, do not heavily penalize it.
- If the assistant asks for clarification when the request was already fully specified, cap the score at 0.50. If it asks for clarification after making an invalid or incomplete tool call, score based on the invalid call, not on the later clarification.

1. Task Completion (PRIMARY)

Did the assistant actually accomplish what the user asked for?

- Judge by the quality and correctness of the tool call, not by whether the mock server returned a result.
- A tool call that returns content but uses wrong parameters (e.g., wrong language, wrong entity, wrong value) is considered a task failure.
- If the user's intent is clear and actionable, the assistant must invoke the appropriate tool with accurate parameters.
- A single tool call can complete a multi-part user request when the tool definition explicitly supports the combined workflow. Do not require multiple calls just because the user phrased the task as multiple actions.
  Example: if `create_album` supports `search_query`, then "search photos for X and create a normal album named Y" can be fully completed by `create_album(album_name=Y, search_query=X, album_type=normal)`.
- If the user's intent is **not sufficiently specified**, then **correctly asking for clarification instead of calling a tool counts as successful handling** of the task and should generally receive a high score.
- If both tasks are completed equally well, the one with fewer turns must receive a higher score.

2. Function Selection Accuracy

Is the selected function appropriate given the user's intent and available tools?

- Selecting a function only loosely related to the user query is incorrect.
- Invoking any function when the user intent is underspecified is an error.
- Prefer the function whose schema directly covers the user's full workflow over a narrower helper function that only completes part of the request.
- If the assistant uses a narrower helper function first and then a valid completing function, judge the final task completion by the valid completing function and its parameters. Do not penalize the extra helper call unless it introduces wrong parameters, wrong language, unsupported claims, or other user-visible harm.

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

12. Response Coherence

Natural language responses and follow-up assistant messages must be coherent, relevant, and in the user's language unless the user asked otherwise.

- Penalize random mixed-language text, unrelated named entities, fabricated search results, fabricated file or study contents, or statements that conflict with the chosen tool call.
- Do not require a verbose confirmation after a successful tool call; a correct, minimal tool call can receive a high score.
- A natural language answer can receive a high score only when no available tool should be called, the assistant explains the limitation or asks the necessary clarification, and the response is coherent.

---

CRITICAL INSTRUCTIONS

- The mock server's success responses are simulated — do NOT use them as evidence of task success.
- The mock server's error responses are real — use them as evidence of parameter or schema problems.
- Judge success based on whether the parameters would achieve the user's actual goal in a real system.
- Tool call syntax format (JSON vs XML) is NOT a scoring criterion.
- Tool call parsability IS a scoring criterion: malformed tags, invalid JSON arguments, missing closing structures, or mixed natural-language text inside structured parameters should be treated as schema/format failures.
- The final score for each trajectory must be a floating-point number between 0 and 1 (inclusive).
- Score each trajectory independently and absolutely.
- Do NOT normalize scores across the batch.
- Do NOT force artificial score differences between similar trajectories.
- If two trajectories are equivalent in quality, assign the same score.

---

Normalized Scoring Guidelines (0–1)

- 0.90 – 1.00 (Excellent)
  - Correct function, all parameters accurate and faithful to user input, schema-compliant, and appropriate clarification for underspecified intent.
  - A supported compound workflow is completed with the most direct valid tool call, including all needed parameters.
  - Also use this range when the request is underspecified and the assistant correctly asks for the missing information in the user's language without hallucinating content.
  - Also use this range when the user requests a clear but unsupported action (for example, NAS shutdown with no shutdown tool available) and the assistant correctly avoids tool calls and explains the limitation in the user's language.
  - The assistant response is coherent and does not contradict, fabricate, or add unrelated content.

- 0.70 – 0.89 (Good)
  - Correct function with minor issues (e.g., one optional parameter mishandled, minor non-critical deviation), or a good clarification response with slight phrasing issues.
  - A compound workflow is completed with valid parameters, but the assistant uses an unnecessary helper call before the completing call.
  - Correct tool call with minor awkward wording in the follow-up response that does not undermine the completed action.

- 0.50 – 0.69 (Acceptable)
  - Correct function selected, but parameter inaccuracies partially undermine task success; or clarification is present but incomplete, awkward, or partially inconsistent.
  - Correct or mostly correct tool call followed by confusing or partially irrelevant text that does not clearly reverse the action.

- 0.30 – 0.49 (Poor)
  - Incorrect function selection, or correct function with parameters that would fail the task in a real system.
  - Valid-looking but materially incomplete tool call, such as choosing the right media type but omitting the required playable title/source.

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
  - Mostly incoherent output, unrelated multilingual fragments, or malformed tool syntax

- 0.00 – 0.10 (Hard Failure)
  - No tool call made when one was clearly required and the intent was unambiguous
  - Assistant claims to have completed an unsupported system-level action, such as shutting down the NAS, without a valid tool call capable of doing so
  - Tool invoked when the user query does not justify an actionable decision
  - Assistant fabricates an answer instead of requesting necessary missing information

---

Output Format

Return a JSON object with a `scores` array. Include one object per trajectory.

{
  "scores": [
    {
      "trajectory_id": "1",
      "score": 0.0,
      "explanation": "Concise explanation covering: (1) whether the task was actually completed given the parameter quality, (2) function selection correctness, (3) parameter faithfulness and completeness with concrete examples, (4) any language consistency or coherence violations, and (5) how ambiguous intent was handled."
    }
  ]
}

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
