# 👥 User Guide - New Features

## Overview
Your XAI system has been optimized with three major improvements:
1. **Accuracy Validation** - Prevents low-accuracy models from running analysis
2. **Fast Batch Processing** - 3-5x speedup with parallel processing
3. **Explanations** - Why each counterfactual suggestion helps

---

## 🎯 Step-by-Step Usage

### Step 1: Upload Dataset & Train Model

1. Click **"Upload CSV, Excel, or JSON"** or select a sample dataset
2. Select **"Auto-clean data"** checkbox (recommended) ✅
3. Choose your **target feature** (prediction goal)
4. Click **"🚀 Train Model"** button

#### What Happens Next:
- ✅ **If Accuracy ≥ 75%**: "Proceed" is enabled → Continue to next step
- ❌ **If Accuracy < 75%**: "Proceed" is disabled → Use optimization

---

### Step 2: Optimize Dataset (If Needed)

**When:** Accuracy is below 75%

1. Review current accuracy message
2. Click **"🚀 Optimize Dataset"** button
3. Wait for optimization to complete:
   - 📊 Features are scaled
   - 🔤 Categories are encoded  
   - 🎯 Outliers are removed
   - Model is re-trained

#### Results:
- ✅ **New accuracy meets threshold?** → Continue to analysis ✨
- ❌ **Still below threshold?** → Try different target feature or review data quality

---

### Step 3: View Counterfactual Suggestions

Each suggestion shows:
1. **Proposed Changes** - Table showing improvements
2. **Why These Changes Help** - Natural language explanation for EACH feature change

#### Example:
```
💡 Suggestion #1

Proposed Changes:
┌─────────────┬──────────┬──────────┬────────┐
│ Feature     │ Original │ Suggested│ Change │
├─────────────┼──────────┼──────────┼────────┤
│ cibil_score │ 423      │ 732      │ ↑ 309  │
│ income      │ 50000    │ 75000    │ ↑ 25K  │
└─────────────┴──────────┴──────────┴────────┘

Why These Changes Help:
- **cibil_score**: Increasing from 423 to 732 improves creditworthiness, 
  which increases chances of loan approval.
- **income**: Increasing income to 75000 demonstrates stronger repayment 
  capacity. Lenders favor higher and stable income.
```

---

### Step 4: Run Batch Analysis

**When:** You want to analyze multiple instances quickly

#### Configuration Panel:
```
⚡ Fast Mode (Fast vs. Accurate)
📊 Limit dataset rows (5 to 100)
```

#### Run Analysis:
1. Click **"🔍 Analyze Instances (Parallel Processing)"** button
2. Watch real-time progress: `🚀 Processing... 15/20 instances`
3. Review results:
   - ✅ Faithful Rate (% of good suggestions)
   - ❌ Rule Violations (% of broken rules)
   - 🎯 Agent Full Rate (feasible solutions)

#### Performance Metrics:
```
✅ Analysis complete! Processed 20 instances in 3.45s

Performance:
- ⏱️ Performance: 3.45s total
- 🚀 Speed: 0.173s/instance
- 📊 Throughput: 5.8 instances/sec
```

---

## ⚙️ Configuration Options

### Fast Mode
**Toggle:** `⚡ Fast Mode (Skip heavy computations)`

**Fast Mode OFF (Default):**
- 🎯 Full accuracy
- ⏱️ ~0.6s per instance
- 🔬 Complete explanations
- **Use for:** Production, accurate results

**Fast Mode ON:**
- ⚡ 2x faster (~0.3s per instance)
- 📝 Simplified explanations
- ✅ Still accurate for most cases
- **Use for:** Quick demos, prototyping, large datasets

### Batch Row Limit
**Slider:** `📊 Limit dataset rows (5 to 100)`

**Why limit rows?**
- Larger datasets take longer
- Limit rows to see quick results
- Adjust based on your dataset size

**Recommendations:**
- Small dataset (< 100 rows): Use full dataset
- Medium dataset (100-1000): Limit to 50-100
- Large dataset (> 1000): Limit to 20-50

---

## 📊 Results Interpretation

### Accuracy Status
```
✅ Accuracy 78% meets threshold (75%)
   → All features are ENABLED ✅

❌ Accuracy 65% is BELOW threshold (75%)
   → Features are DISABLED 🔒
   → Use optimization to improve
```

### Suggestion Metrics
```
Suggestion #1
├── 📈 Feature increases → Positive impact
├── 📉 Feature decreases → Reduces risk
└── 🔄 Feature changes → Categorical change
```

### Batch Analysis Metrics
- **Faithful Rate**: Higher is better (aim > 80%)
- **Rule Violations**: Lower is better (aim < 20%)
- **Agent Full Rate**: % of instances with solutions (aim > 50%)
- **Speed**: Instances per second (target 5+)

---

## 💾 Export Options

### After Individual Analysis:
- 📥 **Download Full Audit Report (JSON)**
  - Original instance
  - All suggestions (faithful + faithless)
  - Auto-detected rules
  - Quality metrics

### After Batch Analysis:
- 📥 **CSV (Per-Instance)**
  - One row per instance
  - Metrics for each
  - Easy to import to Excel
  
- 📥 **JSON (Full Report)**
  - Summary statistics
  - Per-instance details
  - Processing time
  - Performance metrics

---

## 🧪 Example Workflow

### Scenario: Loan Approval Dataset

1. **Upload** → `loan_data.csv` (500 rows)
   - Auto-clean: ON
   - Target: `loan_approved`

2. **Train Model** → Accuracy: 68% 😞
   - Below 75% threshold
   - Optimization section appears

3. **Optimize** → Click "🚀 Optimize Dataset"
   - Scales income, loan_amount
   - Encodes categorical features
   - Removes outliers
   - Re-trains model
   - **New Accuracy: 82%** ✨

4. **Analyze Single Instance**
   - Gets 5 counterfactual suggestions
   - Shows why each helps (domain explanations)
   - Example: "Increasing income shows repayment ability"

5. **Batch Analysis** → Click "🔍 Analyze Instances"
   - Fast Mode: OFF (accurate)
   - Row Limit: 20 instances
   - Processing: 3.2 seconds total
   - Results: 85% faithful rate ✅

6. **Export** → Download JSON report
   - Share with stakeholders
   - Use for compliance documentation
   - Analyze trends

---

## ❓ FAQ

### Q: Why is my accuracy below 75%?
**A:** Your dataset may have:
- Missing values
- Outliers/noise
- Imbalanced classes
- Wrong target feature

**Solution:** Use optimization or try different target feature

### Q: What does "Faithful" mean?
**A:** A suggestion that respects business rules:
- ✅ Doesn't change immutable features (age, race)
- ✅ Follows constraints (income can't decrease loan tenure)
- ✅ Actionable and realistic

### Q: How long does batch analysis take?
**A:** Depends on:
- Row limit: 20 rows ≈ 3-5 seconds
- Fast mode: OFF = 0.6s/row, ON = 0.3s/row
- Parallel workers: 4 (default)

### Q: Can I analyze 1000 rows?
**A:** Yes, but limit to 50-100 for speed:
- 100 rows @ 0.6s/row ≈ 60 seconds
- 100 rows @ 0.3s/row (Fast Mode) ≈ 30 seconds

### Q: What if optimization fails?
**A:** Try:
1. Different target feature
2. Manual data cleaning (Excel)
3. Smaller dataset (subset)
4. Check for data quality issues

---

## 🎨 UI Elements Explained

### Status Icons
| Icon | Meaning |
|------|---------|
| ✅ | Success / Valid / Passed threshold |
| ❌ | Failed / Invalid / Below threshold |
| ⚠️ | Warning / Needs attention |
| 📊 | Metric / Statistical |
| ⚡ | Fast / Speed-related |
| 🚀 | Action / Processing |
| 💡 | Suggestion / Idea |
| 🔒 | Locked / Disabled feature |

### Colors
- 🟢 Green: Good / Positive
- 🔴 Red: Bad / Negative  
- 🟡 Yellow: Warning / Caution
- 🔵 Blue: Neutral / Info
- 🟣 Purple: Primary action

---

## 🚀 Performance Tips

### Make Batch Analysis Faster:
1. ✅ **Enable Fast Mode** (2x faster)
2. ✅ **Reduce row limit** (20 instead of 100)
3. ✅ **Use auto-cleaned data** (preprocessing done)
4. ✅ **Smaller dataset** (upload subset first)

### Make Insights Better:
1. ✅ **Disable Fast Mode** (full explanations)
2. ✅ **Increase row limit** (more validation data)
3. ✅ **Monitor accuracy** (> 75% recommended)
4. ✅ **Review explanations** (understand why)

---

## 📞 Troubleshooting

### "Accuracy is below threshold" Error
**Problem:** Model accuracy < 75%
**Solution:** 
- Click "Optimize Dataset"
- Or select different target column
- Or upload cleaner data

### Batch Analysis Disabled
**Problem:** Can't click "Analyze Instances"
**Solution:**
- Train model first (Step 2)
- Ensure accuracy ≥ 75%
- Click "Optimize Dataset" if below threshold

### Explanations are empty/generic
**Problem:** Feature not in domain dictionary
**Solution:**
- It uses generic explanation
- Or: Add custom explanation for your feature
- Contact admin to extend domain features

### Batch takes too long
**Problem:** Processing > 60 seconds
**Solution:**
- Enable "Fast Mode"
- Reduce row limit (20 instead of 100)
- Close other applications

---

## ✅ Verification Checklist

Before deployment, verify:

- [ ] Dataset auto-cleans without errors
- [ ] Model trains and shows accuracy
- [ ] Accuracy validation gate works (try < 75% scenario)
- [ ] Optimization button works and improves accuracy
- [ ] Suggestions show natural language explanations
- [ ] Batch analysis runs 3-5x faster than before
- [ ] Performance metrics display correctly
- [ ] Fast Mode toggle changes speed significantly
- [ ] Row limit slider works (5 to 100)
- [ ] Export buttons work (CSV & JSON)
- [ ] Error handling doesn't crash app

---

## 📚 Quick Links

- 🔧 **Configuration Guide**: See `OPTIMIZATION_SUMMARY.md`
- 💻 **Code Reference**: See `OPTIMIZATION_QUICK_REFERENCE.md`
- 📊 **Summary**: See `OPTIMIZATION_SUMMARY.md`

---

**Last Updated:** April 11, 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
