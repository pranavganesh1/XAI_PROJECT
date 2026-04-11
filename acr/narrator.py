"""
LLM Narrator - Generates dynamic, instance-specific explanations using Gemini.
Integrated into the ACR Dashboard.
"""

import os
import json
from dotenv import load_dotenv
import hashlib

load_dotenv()


def generate_explanation(instance_id, query_dict, valid_cfs, invalid_cfs, feature_names, model_pred=None, dataset_name="dataset"):
    """
    Generate a UNIQUE, data-driven explanation for a specific instance.
    
    Args:
        instance_id: Row identifier
        query_dict: Original feature values (dict)
        valid_cfs: Valid counterfactuals (list of dicts)
        invalid_cfs: Invalid counterfactuals (list of dicts)
        feature_names: List of feature names
        model_pred: Model's prediction for this instance
        dataset_name: Name of dataset (for context)
    
    Returns:
        explanation: String narrative tailored to this instance
    """
    print(f"\n[DEBUG] Generating explanation for Instance #{instance_id} on {dataset_name}")
    print(f"[DEBUG] Input row: {query_dict}")
    print(f"[DEBUG] Valid CFs: {len(valid_cfs)}, Invalid CFs: {len(invalid_cfs)}")
    
    # Build instance-specific context
    context = _build_instance_context(instance_id, query_dict, valid_cfs, invalid_cfs, feature_names, model_pred, dataset_name)
    
    # Try LLM-based explanation first
    explanation = _generate_llm_narrative(context, instance_id, query_dict, valid_cfs, invalid_cfs)
    
    if not explanation or explanation.startswith("Could not"):
        explanation = _generate_local_narrative(instance_id, query_dict, valid_cfs, invalid_cfs, feature_names)
    
    print(f"[DEBUG] Generated explanation: {explanation[:100]}...")
    return explanation


def _build_instance_context(instance_id, query_dict, valid_cfs, invalid_cfs, feature_names, model_pred, dataset_name):
    """Build rich context for narrative generation."""
    context = f"Instance #{instance_id} from {dataset_name}\n"
    context += f"Model Prediction: {model_pred}\n"
    context += f"Original Profile: {json.dumps({k: v for k, v in query_dict.items()}, default=str)}\n\n"
    
    if invalid_cfs:
        context += f"REJECTED ({len(invalid_cfs)} suggestions):\n"
        for item in invalid_cfs[:5]:  # Limit to 5
            reason = item.get('reason', 'Rule violation') if isinstance(item, dict) else str(item)
            context += f"  • {reason}\n"
    
    if valid_cfs:
        context += f"\nAPPROVED ({len(valid_cfs)} suggestions):\n"
        for idx, cf in enumerate(valid_cfs[:3], 1):  # Limit to 3
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    orig = query_dict.get(f)
                    new = cf[f]
                    changes.append(f"{f}: {orig}→{new}")
            if changes:
                context += f"  Option {idx}: {', '.join(changes)}\n"
    
    return context


def _generate_llm_narrative(context, instance_id, query_dict, valid_cfs, invalid_cfs):
    """Attempt LLM-based explanation (unique per instance)."""
    prompt = f"""You are an AI advisor generating personalized recommendations.

INSTANCE ID: {instance_id}
DATASET CONTEXT:
{context}

TASK: Generate a personalized, data-driven explanation that:
1. References SPECIFIC values from this person's profile
2. Explains WHY suggestions were rejected (if any)
3. Recommends specific actions they CAN take
4. Includes actionable next steps
5. Is 3-4 sentences, friendly but professional

Generate explanation:"""

    # Try Gemini 2.0 Flash (latest)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                generation_config={'temperature': 0.7, 'max_output_tokens': 300}
            )
            return response.text if response.text else None
        except Exception as e:
            print(f"[DEBUG] Gemini Error: {e}")

    # Fallback to NVIDIA (Llama)
    nv_key = os.getenv("NVIDIA_API_KEY")
    if nv_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nv_key)
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[DEBUG] NVIDIA Error: {e}")
    
    return None


def _generate_local_narrative(instance_id, query_dict, valid_cfs, invalid_cfs, feature_names):
    """Generate explanation without LLM (fallback)."""
    parts = []
    
    # Opening with instance-specific data
    key_features = list(query_dict.items())[:3]
    key_str = ", ".join([f"{k}={v}" for k, v in key_features])
    parts.append(f"**Instance #{instance_id}** ({key_str}): ")
    
    if not valid_cfs and not invalid_cfs:
        parts.append("Insufficient data to generate recommendations.")
    elif not valid_cfs and invalid_cfs:
        parts.append(f"All {len(invalid_cfs)} suggestions required changing immutable features (e.g., age, genetics). No actionable paths available.")
    else:
        parts.append(f"Found {len(valid_cfs)} realistic action(s): ")
        for idx, cf in enumerate(valid_cfs[:3], 1):
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    changes.append(f"{f}: {query_dict.get(f)} → {cf[f]}")
            if changes:
                parts.append(f"Option {idx}: {', '.join(changes)}. ")
    
    return "".join(parts)


def evaluate_explanation(explanation, valid_cfs, invalid_cfs):
    """
    Evaluate if an explanation is valid/useful.
    
    Returns:
        is_valid: bool (True if explanation is non-empty and meaningful)
        quality_score: float (0-1)
    """
    if not explanation or len(explanation.strip()) < 10:
        return False, 0.0
    
    # Check if explanation mentions actual data
    has_specificity = any(keyword in explanation.lower() for keyword in ['option', 'adjust', 'recommend', 'instance', '→', '->', ': '])
    
    # Check if it reflects the CF results
    if valid_cfs:
        has_valid_mention = 'action' in explanation.lower() or 'option' in explanation.lower()
    else:
        has_valid_mention = True  # OK if no valid CFs
    
    is_valid = has_specificity and has_valid_mention
    quality = 0.8 if is_valid else 0.2
    
    return is_valid, quality


def get_narrative(query_dict, valid_cfs, invalid_cfs, feature_names):
    """
    Narrate the audit results using NVIDIA (Llama 3.1) or Gemini.
    Falls back to a local template if APIs are unavailable.
    """
    # 1. Build context
    context = f"Person's Profile: {json.dumps(query_dict, default=str)}\n\n"

    if invalid_cfs:
        context += "REJECTED suggestions (impossible/unfaithful):\n"
        for item in invalid_cfs:
            context += f"- {item['reason']}\n"

    if valid_cfs:
        context += "\nAPPROVED suggestions (actionable):\n"
        for cf in valid_cfs:
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    changes.append(f"{f}: {query_dict.get(f)} → {cf[f]}")
            if changes:
                context += f"- {', '.join(changes)}\n"

    prompt = f"""You are a friendly, professional advisor explaining AI-generated suggestions to a user.

CONTEXT:
{context}

RULES:
- Never suggest changing immutable features (age, race, sex, genetics)
- Explain WHY certain suggestions were rejected (they violated causal rules)
- Highlight the actionable steps the user CAN take
- Be encouraging and constructive
- Keep it to 4-5 sentences maximum

Write a clear, helpful explanation for this person:"""

    # 2. Try NVIDIA (Llama 3.1)
    nv_key = os.getenv("NVIDIA_API_KEY")
    if nv_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nv_key)
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"NVIDIA Error: {e}")

    # 3. Try Gemini (Fallback)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")

    # 4. Local Fallback (Final)
    return _local_narrative(query_dict, valid_cfs, invalid_cfs, feature_names)


def _local_narrative(query_dict, valid_cfs, invalid_cfs, feature_names):
    """Fallback: generate a structured narrative without LLM."""
    parts = ["Based on the causal analysis of your profile:\n"]

    if invalid_cfs:
        parts.append(f"**{len(invalid_cfs)} suggestion(s) were discarded** because they tried to change things that cannot be changed (like age or genetics).\n")

    if valid_cfs:
        parts.append(f"**{len(valid_cfs)} actionable suggestion(s) remain.** Here's what you can realistically do:\n")
        for cf in valid_cfs:
            changes = []
            for f in feature_names:
                if f in cf and str(cf[f]) != str(query_dict.get(f)):
                    changes.append(f"adjust **{f}** from {query_dict.get(f)} to {cf[f]}")
            if changes:
                parts.append(f"- {', '.join(changes)}\n")
    else:
        parts.append("Unfortunately, no actionable suggestions could pass the causal audit. All generated options required changing immutable features.\n")

    parts.append("\n*These suggestions respect your immutable characteristics and focus only on factors within your control.*")
    return ''.join(parts)
