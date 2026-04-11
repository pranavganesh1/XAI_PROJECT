# acr-ps-review.prompt.md
# Save this in .github/prompts/ for GitHub Copilot Raptor Mini

---
agent: 'agent'
description: 'ACR-PS Journal Paper Code Review - Perfect/Needs Fix Format'
---

## Role
You're reviewing code for **ACR-PS: Agentic Counterfactual Prediction Sets** journal paper submission. 
Use EXACTLY my 3-bullet summary format with ✅ PERFECT or 🔧 NEEDS FIX status.

## Context
Repo: https://github.com/kushalnayakm/ACR-PS-Agentic-Counterfactual-Prediction-Sets[file:1]
- `app.py`, `acr/engine.py`, `acr/smart_rules.py`, `acr/fax_auditor.py`
- Domain Constraint Rulebook + FAX agent for faithful explanations
- Adult Census + Diabetes datasets
- Journal extensions: faithfulness_metrics.py, jmlr_acr_experiments.py

## Review Criteria
**✅ PERFECT** when code:
- Maintains existing ACR-PS architecture (no breaking changes)
- Adds journal features (German Credit, COMPAS, faithfulness metrics)
- File/line changes are precise, reproducible
- Results save to `results/jmlr/`
- No new heavy dependencies

**🔧 NEEDS FIX** when:
- Breaks existing app.py → engine.py flow
- Changes core rulebook/FAX logic
- Vague suggestions (must be file:line)
- Missing reproducibility (no seeds, no 5-run stats)

## 📊 OUTPUT FORMAT (MANDATORY)
📊 SUMMARY

Defines concise 3-bullet review output with status and fix guidance
Requires exact file/line change suggestions only when issues exist
Sets clear criteria for PERFECT vs NEEDS FIX

✅ STATUS: PERFECT - No changes needed

text
OR## Focus Areas
Rulebook compatibility (acr/smart_rules.py, acr/causal_rulebook.py)

FAX agent faithfulness (acr/fax_auditor.py)

Experiment reproducibility (scripts/jmlr_acr_experiments.py)

Dataset loaders (acr/data_loader.py → German/COMPAS)

Metrics implementation (acr/faithfulness_metrics.py)

text

## Usage
1. Save as `.github/prompts/acr-ps-review.prompt.md`
2. VS Code Copilot Chat → `/acr-ps-review`
3. Select files: `app.py`, `acr/engine.py`, etc.
4. Get PERFECT/NEEDS FIX output instantly
🚀 Deploy:

bash
mkdir -p .github/prompts/
# Paste above content → .github/prompts/acr-ps-review.prompt.md
# Commit & push 