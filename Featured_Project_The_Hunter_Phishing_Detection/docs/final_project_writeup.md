# The Hunter: Final Project Writeup

**ITAI 2376 - Deep Learning in AI**
**Houston City College**
**Team 1:** Chloe Tu, Mattew Choo, Franck Kolontchang
**Date:** April 2026

---

## What Our Agent Does

The Hunter is an email phishing detection system. It takes a raw email, runs it through a trained deep learning classifier, and decides what to do about it — quarantine it, flag it for review, log it, or let it through. The difference between this and a normal classifier is that three AI agents handle the entire process autonomously, passing work to each other the way analysts in a real security operations center would.

Agent 1 (the Ingest Specialist) cleans up the raw email text and pulls out signals like URL count, urgency keywords, and text length. Agent 2 (the Risk Analyst) feeds that cleaned text into the ensemble classifier, gets a phishing probability score, and decides on its own whether the score is confident enough or whether it needs to dig deeper with regex-based pattern analysis. Agent 3 (the SOC Orchestrator) takes the risk assessment, applies a four-tier defense policy, checks a SQLite threat memory database for repeat offenders, and writes the final security verdict.

We tested it on four emails — an obvious phishing email, a clean legitimate one, a borderline case, and a repeat sender — to show that it handles each situation differently and autonomously.

---

## Option Choice and Midterm Upgrade

We chose **Option B: Multi-Agent System.** Our midterm blueprint laid out the full architecture on paper — three agents in a sequential pipeline, a BiLSTM ensemble classifier, five tools, four demo emails, and a persistent threat memory. The blueprint covered the problem statement, the agent architecture diagram, the deep learning connection to Modules 02, 04, and 05, the framework rationale, the tool and dataset inventory, and a six-week build plan.

For the final, we implemented the blueprint end-to-end in a single Colab-ready Jupyter notebook and built a dedicated Gradio web application for real-time defense demonstration. The core architecture — three agents, their roles, the tool assignments, the datasets, the sequential pipeline — stayed the same as what we planned. What changed was everything we discovered while making it actually run, which is most of what follows.

---

## What Changed From the Midterm Blueprint and Why

The blueprint anticipated four challenges: keeping agent tasks goal-oriented rather than scripted, borderline email misclassification, class imbalance in training data, and SQLite persistence across Colab sessions. We addressed all four of those. What we did not anticipate were the infrastructure-level problems that ended up consuming most of our debugging time.

**The LLM was decommissioned.** The blueprint specified Groq's `llama3-8b-8192` model for agent reasoning. Groq deprecated that model during our development window. We had to migrate to `llama-3.1-8b-instant` — a newer version of the same model family — and update every agent definition and configuration reference.

**The tool interfaces had to get simpler.** The blueprint designed the classifier tool to accept multiple input fields: cleaned text, pre-computed features, feature names, and a sender hint. When we built this, Groq's tool-calling validation treated every field as required regardless of our Optional annotations in the Pydantic schema, so every call to the classifier failed. We had to strip the input down to a single field (just the email text) and have the tool compute all derived features internally. The blueprint's architecture diagram shows a richer data hand-off between agents than what survived contact with the API.

**The Groq tier was not enough.** The blueprint listed "Groq API (free tier)" under infrastructure. The free tier provides 6,000 tokens per minute. A single three-agent pipeline consumes roughly 4,000-5,000 tokens, meaning the first email would succeed and every subsequent email would hit the rate limit. The error messages were not helpful — we saw "None or empty response from LLM call" and "Maximum iterations reached" and spent time debugging the agent logic before realizing the root cause was the token budget. We upgraded to the Groq Developer Tier (250,000 tokens per minute, still free) and the problem went away.

**The LLM hallucinated tools that do not exist.** After Agent 1's ingest tool returned a valid result, the LLM would sometimes try to call a tool called `brave_search` — something it picked up from its training data, not from anything we defined. This crashed the pipeline. We fixed it by setting a flag (`result_as_answer`) on the ingest tool so the tool's output immediately becomes the agent's final answer, preventing the LLM from attempting a second action.

**We added guards the blueprint never mentioned.** Retry logic with backoff delays. A 90-second cooldown between emails. Tightened iteration limits (Agent 1 gets two tries, Agents 2 and 3 get three). A wrapper function that catches hallucinated-tool errors and retries the full crew. None of this was in the blueprint because we did not know we would need it. All of it was necessary to get four emails through the pipeline without failures.

**One spec detail changed.** The blueprint said the deep analysis tool could adjust a score by "up to +0.3." In the final implementation it is +0.34. Minor, but worth noting for accuracy.

---

## Agent Framework: CrewAI

We used **CrewAI** as our agent framework, which is what the blueprint specified. The blueprint gave three reasons for choosing it, and all three held up during implementation.

First, CrewAI's role-based agent design kept each agent in its lane. Agent 2 never tried to issue a security verdict (Agent 3's job), and Agent 3 never tried to run the classifier (Agent 2's job). The role, goal, and backstory that you assign each agent go into its system prompt and genuinely constrain its behavior.

Second, CrewAI lets you write task descriptions as goals and expected outputs rather than step-by-step instructions. This is what the blueprint called the key to producing "genuine agentic behavior" — and it worked. Agent 2's task says to produce a confident risk assessment using whatever tools it needs. The decision to escalate to deep analysis on ambiguous scores is made by the LLM's reasoning, not by conditional logic in our code.

Third, CrewAI's Pydantic-based tool schemas caught type errors before they reached the LLM. That said, the interaction between CrewAI's schemas and Groq's schema validation caused its own problems (the Optional fields issue described above), so this benefit came with a cost we did not expect.

We chose CrewAI over LangGraph and AutoGen because its role-based structure mapped directly to our three-agent SOC architecture. Each agent corresponds to a real security operations role, and the sequential pipeline matched our analysis flow naturally.

---

## Deep Learning Models and How They Fit

We integrated two models into an ensemble classifier, exactly as the blueprint described.

**Bidirectional LSTM (BiLSTM).** This is the primary model, connected to Module 04 (Recurrent Neural Networks). It reads email text as a sequence and learns patterns that indicate phishing — urgency phrases, credential requests followed by URLs, suspicious sentence structures. It is a two-layer stacked BiLSTM (128 units into 64 units) with a trainable embedding layer (Module 02 — Representation Learning) that maps each word to a 128-dimensional vector. Dropout layers prevent overfitting. It carries 65% of the ensemble weight.

**Logistic Regression.** This is the secondary model. It works on five features we extract from each email: character count, word count, URL count, exclamation count, and urgent keyword count. It is simpler than the BiLSTM but captures structural signals the text model might miss. It carries 35% of the weight and was trained with SMOTE oversampling to handle the class imbalance in the Alam dataset (about 28,000 emails).

The ensemble combines both models' probability scores using those weights. Instead of a default 0.5 cutoff, we computed the optimal threshold from the Precision-Recall curve on held-out test data. For a security system, missing a real phishing email is worse than flagging a safe one, so the threshold is set with that priority.

We also evaluated a standalone Random Forest on the Cratchley dataset (500,000+ emails with header and metadata features) to show that phishing signals exist beyond the email body. That model is not wired into the agent pipeline because its features — things like domain age and DNS records — are not available from raw email text at runtime. The blueprint explained this separation, and we kept it.

The BiLSTM ensemble connects to the agent system through Agent 2's classifier tool. When Agent 2 calls the tool, it runs the ensemble and returns a phishing probability score. Agent 2 then reasons about the score and decides whether to pass it forward or escalate to deeper analysis. That reasoning comes from the Transformer powering the agent (LLaMA 3.1 8B via Groq — Module 05), not from any if-statement in our code. The blueprint called this the "architectural division of labor": the BiLSTM does deep sequence classification, and the Transformer does reasoning and tool orchestration. Both are essential and neither could replace the other.

---

## What Worked, What Did Not, and What We Would Improve

**What worked.** The three-agent separation of concerns worked as designed. Each agent stayed focused on its job. Agent 2 autonomously escalating to deep analysis on a borderline email — without us writing conditional logic to tell it to — was the clearest proof that the system is actually agentic and not just a script. The PR-curve threshold calibration gave us a principled decision boundary. The threat memory system correctly flagged a repeat sender on the fourth demo email. All four of the blueprint's anticipated challenges were addressed successfully.

**What did not work.** Getting the pipeline to run reliably was harder than building the models, and the problems were not the ones we anticipated.

- Groq deprecated our target LLM mid-development. We had to migrate to a new model and retest everything.
- The LLM hallucinated calls to tools that do not exist, crashing the pipeline until we added architectural guards to prevent it.
- Groq's schema validation rejected our tool interfaces in ways that are not documented, forcing us to simplify the data contracts between agents.
- The free-tier rate limit was too low for multi-agent workloads, and the resulting error messages pointed in the wrong direction. We spent time debugging agent logic before diagnosing the real cause.
- Agents given too many iterations would loop and produce garbage output. Tightening iteration limits and adding retry wrappers stabilized the system.

The common theme is that the agent design itself was sound. What broke was the layer between the agents and the LLM provider. Most of our debugging time went into problems that had nothing to do with phishing detection and everything to do with API behavior, rate limits, and LLM unpredictability.

**What we would improve with more time.** Train on a larger and more diverse email dataset — 28,000 emails from one source is limited and domain shift to enterprise email styles would likely degrade accuracy. Add email header and DNS metadata features to the pipeline, drawing on the Cratchley-style signals we evaluated separately. Implement campaign-level detection so the system can recognize coordinated phishing attacks targeting multiple recipients. While we successfully upgraded our deliverable from a notebook demonstration to a real-time deployed Gradio frontend service, in the future we would replace the frontend with direct webhook integrations into enterprise email servers.

---

## One Thing That Surprised Us About AI Agents

The thing that surprised us most is how much of the work is not about the AI. We spent more time debugging rate limits, schema validation errors, model deprecations, and hallucination guards than we spent on the actual agent design or the deep learning models. The agents themselves, once properly constrained, did their jobs. But the gap between "this works on paper" and "this runs four emails without crashing" was much wider than the blueprint suggested.

The blueprint's anticipated challenges section listed four things we thought would be hard: goal-oriented task writing, borderline misclassification, class imbalance, and SQLite persistence. We solved all four without too much trouble. The things that actually consumed our time — a deprecated model, hallucinated tool calls, undocumented API validation rules, and a rate limit that silently starved the agents — were not on the list at all. Building an agent system right now is less about designing the agent logic and more about defensive engineering around everything the LLM and its API can do to break your system.
